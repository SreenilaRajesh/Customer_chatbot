import os
import asyncio
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

input_folder = r"C:\Users\visah\Documents\GitHub\Autodesk\Autodesk\pages"  # Folder containing HTML files
output_folder = r"C:\Users\visah\Documents\GitHub\Autodesk_Chatbot\markdown_files_crawler"  # Folder to save Markdown files

async def convert_html_to_md(crawler, html_file):
    file_url = f"file://{html_file}"
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    result = await crawler.arun(url=file_url, config=config)
    if result.success:
        md_filename = os.path.splitext(os.path.basename(html_file))[0] + ".md"
        md_path = os.path.join(output_folder, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)
        print(f"Converted: {html_file} → {md_path}")
    else:
        print(f"Failed: {html_file} | {result.error_message}")

async def crawl_all_html():
    async with AsyncWebCrawler() as crawler:
        tasks = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".html"):
                html_path = os.path.join(input_folder, filename)
                tasks.append(convert_html_to_md(crawler, html_path))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(crawl_all_html())