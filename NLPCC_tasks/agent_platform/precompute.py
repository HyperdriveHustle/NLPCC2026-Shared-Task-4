#!/usr/bin/env python3
"""
预跑新闻摘要和舆情分析，不依赖回测引擎。
结果保存到 cache/ 目录，后续回测直接读取复用。

用法：
  # 本地 Qwen 服务
  python precompute.py --api-base http://your-server:port/v1 --api-key your-key --model qwen-plus

  # 只跑新闻摘要（快）
  python precompute.py --api-base ... --task news

  # 跑新闻 + 舆情分析
  python precompute.py --api-base ... --task all

  # 指定数据范围
  python precompute.py --api-base ... --start-date 2025-01-01 --end-date 2025-12-31
"""

import argparse
import asyncio
import csv
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

# ─── 配置 ───
NEWS_SOURCES = ["caixin", "sina_finance", "tencent_stock", "tiantian_fund"]
NEWS_FILE_PATTERN = "{source}_daily_dedup.csv"
TOP_RANK = 20
MAX_CONCURRENCY = 10  # 并发数，根据服务器调整
SAVE_INTERVAL = 50    # 每 N 条保存一次

SUMMARY_PROMPT = """请将以下金融新闻提取为非常简短的摘要（1-2句话），只保留核心信息：

**原始新闻**:
时间：{date}
标题: {title}
来源: {source}
排名: {ranking}
内容: {content}

**要求**:
1. 提取最核心的市场影响信息
2. 用1-2句话概括，非常简短精炼
3. 如果是无关的市场噪音，返回"无关键信息"
4. 只返回摘要内容，不要额外解释

**摘要**:
"""

# ─── 新闻摘要缓存 ───
news_cache = {}
cache_file = None  # 在 main 中设置
dirty_counter = 0


def _get_news_summary(n: Dict) -> str:
    """从缓存获取新闻摘要"""
    key = f"{n['THEDATE']}_{n['TITLE']}_{n.get('APP_TYPE', '')}_{n.get('RANKING', '')}"
    return news_cache.get(key, "无摘要")


def _compute_hash(data: str) -> str:
    return hashlib.md5(data.encode("utf-8")).hexdigest()[:12]


def load_cache(filepath: str) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  [Cache] 已加载 {len(data):,} 条历史摘要")
            return data
        except Exception:
            pass
    return {}


def save_cache(data: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, filepath)
    print(f"  [Cache] 已保存 {len(data):,} 条到 {filepath}")


# ─── 数据加载 ───
def load_news_items(news_data_dir: str, start_date: str, end_date: str) -> List[Dict]:
    """加载指定日期范围内的新闻，过滤 top-rank"""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    all_items = []
    for source in NEWS_SOURCES:
        filename = NEWS_FILE_PATTERN.format(source=source)
        # 先找 export_data，再找 demo
        for subdir in ["export_data", "2025_12_news_demo"]:
            filepath = os.path.join(news_data_dir, subdir, filename)
            if os.path.exists(filepath):
                break
        else:
            print(f"  [警告] 找不到新闻文件: {source}")
            continue

        print(f"  [加载] {source}...")
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ranking = int(row.get("RANKING", 999))
                if ranking > TOP_RANK:
                    continue

                thedate_str = row.get("THEDATE", "")
                try:
                    thedate = datetime.strptime(thedate_str.split()[0], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue

                if thedate < start_dt or thedate > end_dt:
                    continue

                all_items.append({
                    "THEDATE": thedate_str,
                    "TITLE": row.get("TITLE", ""),
                    "APP_TYPE": row.get("APP_TYPE", source),
                    "RANKING": ranking,
                    "CONTENT": row.get("CONTENT") or "",
                })
                count += 1

        print(f"    找到 {count:,} 条 (top-{TOP_RANK}, {start_date}~{end_date})")

    # 按 ranking 排序
    all_items.sort(key=lambda x: x.get("RANKING", 999))
    print(f"  [合计] {len(all_items):,} 条新闻待处理")
    return all_items


# ─── LLM 调用 ───
async def summarize_news(item: Dict, llm) -> Optional[str]:
    """调用 LLM 生成单条新闻摘要"""
    content = item.get("CONTENT", "") or ""
    prompt = SUMMARY_PROMPT.format(
        date=item.get("THEDATE", "无日期"),
        title=item.get("TITLE", "无标题"),
        source=item.get("APP_TYPE", "未知"),
        ranking=item.get("RANKING", "N/A"),
        content=content[:500],  # 截断过长内容
    )

    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=120)
        if response and hasattr(response, "content") and response.content:
            summary = response.content.strip()
            summary = re.sub(r'["\']', "", summary)
            if len(summary) > 500:
                summary = summary[:497] + "..."
            return summary
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        print(f"    [错误] {item.get('TITLE', '?')[:30]}: {e}")
    return None


async def process_news_batch(items: List[Dict], llm) -> Dict:
    """批量处理新闻摘要，带缓存和进度"""
    global dirty_counter

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def sem_task(item):
        async with semaphore:
            cache_key = f"{item['THEDATE']}_{item['TITLE']}_{item['APP_TYPE']}_{item['RANKING']}"

            if cache_key in news_cache:
                return cache_key, news_cache[cache_key], True  # cache hit

            summary = await summarize_news(item, llm)
            if summary:
                news_cache[cache_key] = summary
                return cache_key, summary, False
            return cache_key, None, False

    tasks = [sem_task(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    hits = 0
    misses = 0
    errors = 0
    for result in results:
        if isinstance(result, Exception):
            errors += 1
        elif result:
            key, summary, is_hit = result
            if is_hit:
                hits += 1
            elif summary:
                misses += 1
            else:
                errors += 1

    dirty_counter += misses
    if dirty_counter >= SAVE_INTERVAL:
        save_cache(news_cache, cache_file)
        dirty_counter = 0

    return {"hits": hits, "misses": misses, "errors": errors}


# ─── 舆情分析 ───
async def run_sentiment_analysis(
    daily_groups: Dict[str, List[Dict]],
    fund_pool: List[str],
    llm,
    output_file: str,
):
    """对每天的新闻进行舆情分析"""
    from agent_platform.agents.fund_info import FUND_INFO
    from agent_platform.utils import CustomJsonOutputParser

    parser = CustomJsonOutputParser()
    sentiment_cache_file = cache_file.replace("news", "sentiment")
    sentiment_cache = load_cache(sentiment_cache_file)

    funds_text = "\n".join([
        f"- {fund} ({FUND_INFO.get(fund, {}).get('name', 'Unknown')}): "
        f"{FUND_INFO.get(fund, {}).get('scope', 'N/A')}。"
        f" ({FUND_INFO.get(fund, {}).get('meaning', 'Unknown')})"
        for fund in fund_pool
    ])

    output_formatter = {
        "overall_sentiment": "positive/negative/neutral",
        "fund_analysis": {
            "基金代码": {
                "sentiment": "positive/negative/neutral",
                "reason": "简要原因",
                "confidence": 0.8,
            }
        },
        "summary": "整体市场舆情摘要",
    }

    total_days = len(daily_groups)
    processed = 0
    skipped = 0

    for date_str, news_items in sorted(daily_groups.items()):
        cache_key = f"{date_str}_{_compute_hash(json.dumps(sorted([(n['THEDATE'], n['TITLE'], n.get('APP_TYPE', '')) for n in news_items])))}"
        if cache_key in sentiment_cache:
            skipped += 1
            processed += 1
            print(f"  [{processed}/{total_days}] {date_str} [缓存命中]")
            continue

        news_text = "\n\n".join([
            f"{n.get('THEDATE', '')} 【{n.get('APP_TYPE', '')}】排名{n.get('RANKING', '')}: "
            f"{n.get('TITLE', '')}\n摘要: {_get_news_summary(n)}"
            for n in news_items
        ])

        prompt = f"""你是一个专业的金融市场舆情分析师。请分析以下新闻对指定投资基金的影响，用于后续生成今日的交易指令，今天是{date_str}：

**可投资基金列表**:
{funds_text}

**处理后的新闻摘要**（共{len(news_items)}条）:
{news_text}

**分析要求**:
1. 判断新闻对哪些基金可能有正面/负面/中性影响
2. 排除完全无关的新闻内容
3. 对每个基金，给出整体的舆情判断（positive/negative/neutral）
4. 提供简要的市场舆情状态和预测
5. 如果没有相关信息，明确说明"无相关信息"

**输出格式**（严格的JSON）:
{json.dumps(output_formatter, ensure_ascii=False)}

请给出你的输出：
"""
        try:
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=300)
            analysis = parser.parse(response.content)
            sentiment_cache[cache_key] = analysis
            processed += 1
            print(f"  [{processed}/{total_days}] {date_str} [完成]")

            if processed % 10 == 0:
                save_cache(sentiment_cache, sentiment_cache_file)
        except Exception as e:
            print(f"  [{processed}/{total_days}] {date_str} [错误: {e}]")
            processed += 1

    save_cache(sentiment_cache, sentiment_cache_file)
    print(f"\n  舆情分析: {processed} 天完成, {skipped} 天缓存命中")


# ─── 主入口 ───
async def main():
    parser = argparse.ArgumentParser(description="预跑新闻摘要和舆情分析")
    parser.add_argument("--api-base", required=True, help="OpenAI 兼容 API base URL")
    parser.add_argument("--api-key", required=True, help="API Key")
    parser.add_argument("--model", default="qwen-plus", help="模型名称")
    parser.add_argument("--task", choices=["news", "all"], default="news",
                        help="news=仅新闻摘要, all=新闻+舆情分析")
    parser.add_argument("--start-date", default="2025-01-01", help="起始日期")
    parser.add_argument("--end-date", default="2025-12-31", help="结束日期")
    parser.add_argument("--news-dir", default=None, help="新闻数据目录（默认自动查找）")
    parser.add_argument("--cache-file", default=None, help="缓存文件路径")
    parser.add_argument("--track", choices=["macro", "sector"], default="macro",
                        help="赛道（影响舆情分析的 fund_pool）")
    parser.add_argument("--max-workers", type=int, default=10, help="最大并发数")
    args = parser.parse_args()

    global cache_file, MAX_CONCURRENCY
    MAX_CONCURRENCY = args.max_workers

    # 查找新闻数据目录
    if args.news_dir:
        news_dir = args.news_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        news_dir = os.path.join(script_dir, "..", "dataset", "news_data")
        if not os.path.exists(news_dir):
            news_dir = os.path.join(script_dir, "dataset", "news_data")
        if not os.path.exists(news_dir):
            print("错误：找不到新闻数据目录，请用 --news-dir 指定")
            sys.exit(1)

    # 缓存文件路径
    if args.cache_file:
        cache_file = args.cache_file
    else:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "news_summaries.json")

    # 加载已有缓存
    news_cache.update(load_cache(cache_file))

    # 初始化 LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        model=args.model,
        temperature=0.1,
    )

    print(f"\n{'='*60}")
    print(f"预跑任务: {args.task}")
    print(f"API: {args.api_base} | 模型: {args.model}")
    print(f"日期: {args.start_date} ~ {args.end_date}")
    print(f"缓存: {cache_file}")
    print(f"{'='*60}\n")

    # 加载新闻
    print("[1/3] 加载新闻数据...")
    news_items = load_news_items(news_dir, args.start_date, args.end_date)
    if not news_items:
        print("没有新闻数据，退出")
        return

    # 过滤已有缓存
    uncached = []
    for item in news_items:
        cache_key = f"{item['THEDATE']}_{item['TITLE']}_{item['APP_TYPE']}_{item['RANKING']}"
        if cache_key not in news_cache:
            uncached.append(item)

    print(f"  总计: {len(news_items):,} 条 | 已有缓存: {len(news_items) - len(uncached):,} | 待处理: {len(uncached):,}\n")

    if uncached:
        # 处理新闻摘要
        print("[2/3] 处理新闻摘要...")
        result = await process_news_batch(uncached, llm)
        print(f"  完成: 命中 {result['hits']:,} | 新增 {result['misses']:,} | 错误 {result['errors']:,}")

        # 最终保存
        save_cache(news_cache, cache_file)
    else:
        print("[2/3] 全部命中缓存，跳过新闻摘要\n")

    # 舆情分析（可选）
    if args.task == "all":
        print("\n[3/3] 舆情分析...")
        # 按日期分组
        daily_groups = {}
        for item in news_items:
            date_str = item.get("THEDATE", "").split()[0]
            daily_groups.setdefault(date_str, []).append(item)

        fund_pool = None
        if args.track == "macro":
            fund_pool = [
                "000300.SH", "000905.SH", "399006.SZ", "000688.SH",
                "000932.SH", "000941.SH", "399971.SZ", "000819.SH",
                "000928.SH", "000012.SH", "518880.SH",
            ]
        else:
            fund_pool = [
                "512880.SH", "512800.SH", "512070.SH", "159995.SZ",
                "159819.SZ", "515880.SH", "159852.SZ", "512010.SH",
                "512170.SH", "159992.SZ", "515170.SH", "512690.SH",
                "512400.SH", "515220.SH", "159870.SZ", "512200.SH",
            ]

        await run_sentiment_analysis(daily_groups, fund_pool, llm, cache_file)

    print(f"\n{'='*60}")
    print(f"预跑完成！缓存已保存到: {cache_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
