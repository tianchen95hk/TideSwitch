import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from config.settings import Settings
from core_data.loader import DataLoader
from core_strategy.algo import WhiteboxAlgo
from core_backtest.engine import BacktestEngine
import os

mpl.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

def main():
    # ==========================================
    # 1. 智能数据检查 (Smart Data Check)
    # ==========================================
    need_download = False
    
    if not os.path.exists(Settings.FACTOR_FILE):
        print("⚡️ 数据文件不存在，准备下载...")
        need_download = True
    else:
        try:
            df_check = pd.read_parquet(Settings.FACTOR_FILE)
            print(f"ℹ️ 检测到现有数据: {len(df_check)} 行")
            
            if 'BTCUSDT' in df_check.index.get_level_values('symbol'):
                btc_len = len(df_check.xs('BTCUSDT', level='symbol'))
                print(f"   BTC 数据长度: {btc_len}")
                
                # [修改] 阈值提高到 5000 (因为我们要 6000 条数据)
                # 确保不会因为意外保留了旧的 1500 条数据而导致回测不完整
                if btc_len < 5000: 
                    print("⚠️ 数据量过少 (可能是旧文件)，强制重新下载！")
                    need_download = True
            else:
                need_download = True
                
        except Exception as e:
            print(f"⚠️ 数据文件损坏，重新下载: {e}")
            need_download = True

    if need_download:
        if os.path.exists(Settings.FACTOR_FILE):
            os.remove(Settings.FACTOR_FILE)
        
        # 使用模块方式调用，防止路径报错
        # 或者确保 loader.py 里的 import 是正确的
        loader = DataLoader()
        loader.run_etl()
    
    # ==========================================
    # 2. 加载与回测
    # ==========================================
    print("📥 读取数据...")
    df = pd.read_parquet(Settings.FACTOR_FILE)
    
    print(f"📅 数据起始时间: {df.index.get_level_values('timestamp').min()}")
    
    strategy = WhiteboxAlgo()
    engine = BacktestEngine(df, strategy)
    res = engine.run()
    
    # ==========================================
    # 3. 绘图与保存 (包含 Benchmark 对比)
    # ==========================================
    print("-" * 30)
    final_equity = res['equity'].iloc[-1]
    ret = (final_equity / Settings.INITIAL_CAPITAL) - 1
    print(f"📊 最终净值: ${final_equity:,.2f} | 收益率: {ret:.2%}")
    
    if not os.path.exists(Settings.OUTPUT_PATH):
        os.makedirs(Settings.OUTPUT_PATH)

    # --- 绘图逻辑优化 ---
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 1. 绘制策略净值
    ax.plot(res.index, res['equity'], label='4Alpha Strategy', color='#1f77b4', linewidth=2)
    
    # 2. 绘制 BTC 基准 (Benchmark)
    # 提取 BTC 价格并截取回测同期
    btc_prices = df.xs('BTCUSDT', level='symbol')['close']
    # 确保索引对齐
    btc_bench = btc_prices.loc[res.index[0]:res.index[-1]]
    # 归一化：让 BTC 从 10万 起跑，以便直观对比
    btc_bench = btc_bench / btc_bench.iloc[0] * Settings.INITIAL_CAPITAL
    
    ax.plot(btc_bench.index, btc_bench, label='BTC Buy & Hold', color='gray', linestyle='--', alpha=0.6)
    
    # 3. 标记买卖点
    if hasattr(engine, 'trade_signals') and engine.trade_signals:
        signals_in_range = [s for s in engine.trade_signals if s.get('timestamp') in res.index]
        signals_in_range = sorted(signals_in_range, key=lambda x: x['timestamp'])

        action_style = {
            'buy':   {'facecolor': 'lightgreen', 'edgecolor': 'darkgreen', 'textcolor': 'darkgreen'},
            'sell':  {'facecolor': 'lightcoral', 'edgecolor': 'darkred', 'textcolor': 'darkred'},
            'short': {'facecolor': 'thistle', 'edgecolor': 'indigo', 'textcolor': 'indigo'},
            'cover': {'facecolor': 'moccasin', 'edgecolor': 'darkorange', 'textcolor': 'darkorange'},
        }

        trade_lines = []
        for idx, s in enumerate(signals_in_range, start=1):
            ts = s['timestamp']
            equity_at = float(res.loc[ts, 'equity'])
            action = s.get('action', '')
            reason = s.get('reason', '')
            delta_txt = s.get('delta_summary', '')

            st = action_style.get(action, {'facecolor': 'white', 'edgecolor': 'black', 'textcolor': 'black'})

            ax.text(
                ts,
                equity_at,
                str(idx),
                fontsize=9,
                fontweight='bold',
                color=st['textcolor'],
                ha='center',
                va='center',
                zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=st['facecolor'], edgecolor=st['edgecolor'], alpha=0.95),
            )

            ts_str = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)
            line = f"{idx:>2} {action.upper():<5} {ts_str} | {reason}"
            if delta_txt:
                line = f"{line} | {delta_txt}"
            trade_lines.append(line)

        max_lines = 28
        if len(trade_lines) > max_lines:
            trade_lines = trade_lines[:max_lines] + [f"... ({len(signals_in_range) - max_lines} more) ..."]

        if trade_lines:
            trade_block = "\n".join(trade_lines)
            fig.text(
                0.72,
                0.5,
                trade_block,
                va='center',
                ha='left',
                fontsize=8,
                family='sans-serif',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9),
            )
    
    ax.set_title(f'4Alpha Strategy vs BTC (Core-Satellite + StopLoss)\nFinal: ${final_equity:,.0f} ({ret:+.2%})', fontsize=14)
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
    
    # 调整布局，为标注留出空间
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.1, right=0.70)
    
    save_path = os.path.join(Settings.OUTPUT_PATH, 'final_result_comparison.png')
    fig.savefig(save_path)
    print(f"✅ 对比图表已保存至: {save_path}")

if __name__ == "__main__":
    main()