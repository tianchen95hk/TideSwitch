import ccxt
import pandas as pd
import time
import os
import math
from config.settings import Settings

class DataLoader:
    def __init__(self):
        options = {
            'defaultType': 'future',
            'timeout': 30000,
            'enableRateLimit': True
        }
        if Settings.USE_PROXY:
            options['proxies'] = {'http': Settings.PROXY_URL, 'https': Settings.PROXY_URL}
            
        self.exchange = ccxt.binance(options)
        self._markets_loaded = False
        if Settings.API_KEY:
            self.exchange.apiKey = Settings.API_KEY
            self.exchange.secret = Settings.SECRET_KEY

    def ensure_markets_loaded(self):
        if self._markets_loaded:
            return
        self.exchange.load_markets()
        self._markets_loaded = True

    def fetch_funding_rate_safe(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('fundingRate', 0.0001))
        except:
            return 0.0001

    def fetch_funding_history_series(self, symbol, start_ts, end_ts, target_index):
        """
        下载 funding 历史序列并对齐到 4H K 线时间戳。
        """
        try:
            self.ensure_markets_loaded()
            market = self.exchange.market(symbol)
            symbol_id = market['id']
        except Exception:
            return pd.Series(index=target_index, data=0.0001, dtype=float)

        start_ms = int(pd.Timestamp(start_ts).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end_ts).timestamp() * 1000)
        cursor = start_ms
        records = []
        page = 0
        max_pages = max(int(Settings.FUNDING_HISTORY_MAX_PAGES), 1)

        while cursor <= end_ms and page < max_pages:
            page += 1
            try:
                batch = self.exchange.fapiPublicGetFundingRate({
                    'symbol': symbol_id,
                    'startTime': cursor,
                    'endTime': end_ms,
                    'limit': 1000
                })
            except Exception:
                break

            if not batch:
                break

            records.extend(batch)
            last_ts = int(batch[-1].get('fundingTime', 0) or 0)
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            time.sleep(0.15)

        if not records:
            return pd.Series(index=target_index, data=self.fetch_funding_rate_safe(symbol), dtype=float)

        fr_df = pd.DataFrame(records)
        if fr_df.empty or ('fundingTime' not in fr_df.columns):
            return pd.Series(index=target_index, data=self.fetch_funding_rate_safe(symbol), dtype=float)

        fr_df['timestamp'] = pd.to_datetime(pd.to_numeric(fr_df['fundingTime'], errors='coerce'), unit='ms', utc=False)
        fr_df['fundingRate'] = pd.to_numeric(fr_df.get('fundingRate', 0.0001), errors='coerce')
        fr_df = fr_df[['timestamp', 'fundingRate']].dropna().drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        if fr_df.empty:
            return pd.Series(index=target_index, data=self.fetch_funding_rate_safe(symbol), dtype=float)

        target_df = pd.DataFrame({'timestamp': pd.to_datetime(target_index)})
        merged = pd.merge_asof(
            target_df.sort_values('timestamp'),
            fr_df,
            on='timestamp',
            direction='backward'
        )
        merged['fundingRate'] = merged['fundingRate'].ffill().bfill().fillna(self.fetch_funding_rate_safe(symbol))
        aligned = pd.Series(index=target_df.sort_values('timestamp')['timestamp'].values, data=merged['fundingRate'].values, dtype=float)
        aligned = aligned.reindex(pd.to_datetime(target_index))
        aligned = aligned.fillna(method='ffill').fillna(method='bfill').fillna(self.fetch_funding_rate_safe(symbol))
        aligned.index = target_index
        return aligned

    def fetch_data_pagination(self, symbol, total_limit):
        """
        🔄 强力循环下载
        """
        all_ohlcv = []
        since_timestamp = None 
        # 每页最大1000条
        page_size = 1000
        # 计算需要几页
        total_pages = math.ceil(total_limit / page_size)
        
        print(f"   📥 {symbol} 准备下载 {total_limit} 条 (约 {total_pages} 页)...")

        for page in range(total_pages):
            try:
                # 计算本次需要多少
                remaining = total_limit - len(all_ohlcv)
                if remaining <= 0: break
                
                limit = min(remaining, page_size)
                
                params = {'limit': limit}
                if since_timestamp:
                    params['endTime'] = since_timestamp

                self.ensure_markets_loaded()
                market = self.exchange.market(symbol)
                data = self.exchange.fapiPublicGetKlines({
                    'symbol': market['id'],
                    'interval': Settings.TIMEFRAME,
                    **params
                })
                
                if not data: 
                    print(f"      ⚠️ 第 {page+1} 页为空，停止。")
                    break
                
                # 转换数据
                # API返回是时间正序: [旧 -> 新]
                # 当我们用 endTime 往前查时，拿到的是更旧的一段
                # 所以要把新拿到的这一段，拼在总列表的最前面
                # batch: [Oct, Nov, Dec]
                # all: [Jan...Sep] + [Oct...Dec] -> 错！
                # 正确逻辑：
                # Round 1 (最新): 拿到 [Oct, Nov, Dec]。 endTime 设为 Oct前一秒
                # Round 2 (次新): 拿到 [Jul, Aug, Sep]。 
                # 拼接应为: Round 2 + Round 1
                
                batch_data = [
                    [float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] 
                    for x in data
                ]
                
                # 拼接到最前面
                all_ohlcv = batch_data + all_ohlcv
                
                # 更新游标 (取这批数据最早的一根K线时间 - 1ms)
                first_ts = int(batch_data[0][0])
                since_timestamp = first_ts - 1
                
                print(f"      ✅ 第 {page+1}/{total_pages} 页完成 | 累计: {len(all_ohlcv)} | 最早时间: {pd.to_datetime(first_ts, unit='ms')}")
                
                time.sleep(0.3) 
                
            except Exception as e:
                print(f"      ❌ 下载中断: {e}")
                break
                
        # 生成 DataFrame
        if not all_ohlcv: return pd.DataFrame()
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        # funding 使用历史序列而非单点快照，增强 carry 因子有效性
        try:
            funding_series = self.fetch_funding_history_series(
                symbol=symbol,
                start_ts=df.index.min(),
                end_ts=df.index.max(),
                target_index=df.index
            )
            df['fundingRate'] = pd.to_numeric(funding_series, errors='coerce').fillna(self.fetch_funding_rate_safe(symbol))
        except Exception:
            df['fundingRate'] = self.fetch_funding_rate_safe(symbol)

        return df

    def run_etl(self):
        print("🔄 初始化市场...")
        self.ensure_markets_loaded()
        
        markets = [m for m in self.exchange.markets.values() if m['linear'] and m['quote'] == 'USDT']
        markets.sort(key=lambda x: float(x['info'].get('quoteVolume', 0)), reverse=True)
        top_symbols = [m['symbol'] for m in markets[:Settings.target_symbols_count]]
        
        if not any('BTC' in s for s in top_symbols):
             top_symbols.insert(0, 'BTC/USDT:USDT')
             
        panel = {}
        print(f"🚀 开始深度下载 (目标: {Settings.LIMIT} 条)...")
        
        for sym in top_symbols:
            clean_name = sym.split(':')[0].replace('/', '')
            df = self.fetch_data_pagination(sym, total_limit=Settings.LIMIT)
            if not df.empty:
                panel[clean_name] = df
                
        print(f"\n💾 保存 {len(panel)} 个币种数据...")
        full_df = pd.concat(panel, names=['symbol'])
        full_df = full_df.swaplevel(0, 1).sort_index()
        
        os.makedirs(os.path.dirname(Settings.FACTOR_FILE), exist_ok=True)
        full_df.to_parquet(Settings.FACTOR_FILE)
        print("✅ 数据下载完成！")

if __name__ == "__main__":
    loader = DataLoader()
    loader.run_etl()
