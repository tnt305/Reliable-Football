import asyncio
import aiohttp
import os
import time
from tqdm.asyncio import tqdm
from typing import List, Tuple
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader


class AsyncSoccerNetDownloader:
    """Wrapper để tải SoccerNet files với asyncio."""
    
    def __init__(self, local_dir: str, password: str, max_concurrent: int = 64):
        self.local_dir = local_dir
        self.password = password
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        os.makedirs(local_dir, exist_ok=True)
    
    async def download_file_async(self, file: str, split: str) -> Tuple[str, bool]:
        """
        Tải file một cách async (chạy trong thread pool).
        Sử dụng asyncio để tránh blocking main thread.
        """
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            try:
                # Chạy blocking operation trong thread pool
                await loop.run_in_executor(
                    None, 
                    self._download_sync, 
                    file, 
                    split
                )
                return (file, True)
            except Exception as e:
                return (file, False, str(e))
    
    def _download_sync(self, file: str, split: str):
        """Hàm đồng bộ thực tế tải file."""
        downloader = SoccerNetDownloader(LocalDirectory=self.local_dir)
        downloader.password = self.password
        downloader.downloadGames(files=[file], split=[split])
    
    async def download_all(self, files: List[str], split: str):
        """Tải tất cả files một cách song song."""
        tasks = [
            self.download_file_async(file, split) 
            for file in files
        ]
        
        results = await tqdm.gather(*tasks, desc="Downloading files", unit="file")
        return results


async def main():
    splits = ["train", "valid", "test"]
    for split in splits:
        files_to_download = [
            "1_baidu_soccer_embeddings.npy",
            "2_baidu_soccer_embeddings.npy",
            "1_224p.mkv",
            "2_224p.mkv",
            "Labels-v2.json",
        ]
        
        local_dir = "./dataset/"
        password = "s0cc3rn3t"
        
        # Tạo downloader với max_concurrent tối ưu
        max_workers = min(64, (os.cpu_count() or 4) * 4)
        downloader = AsyncSoccerNetDownloader(local_dir, password, max_concurrent=max_workers)
        
        print(f"🚀 Bắt đầu tải {len(files_to_download)} files")
        print(f"   Max concurrent: {max_workers}")
        print(f"   Split: {split}\n")
        
        start_time = time.time()
        
        # Tải tất cả files
        results = await downloader.download_all(files_to_download, split)
        
        elapsed = time.time() - start_time
        
        # In kết quả
        print("\n" + "="*50)
        success = sum(1 for r in results if len(r) > 1 and r[1])
        failed = len(results) - success
        
        print(f"✓ Thành công: {success}")
        print(f"✗ Thất bại: {failed}")
        print(f"⏱️  Thời gian: {elapsed:.2f}s")
        print(f"📊 Trung bình: {elapsed/len(results):.2f}s/file")
        
        # Chi tiết từng file
        for result in results:
            if result[1]:
                print(f"  ✓ {result[0]}")
            else:
                print(f"  ✗ {result[0]}: {result[2] if len(result) > 2 else 'Unknown error'}")


if __name__ == "__main__":
    asyncio.run(main())

# import SoccerNet
# from SoccerNet.Downloader import SoccerNetDownloader
# mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/home/storage/thiendc/soccernet")
# mySoccerNetDownloader.password = 's0cc3rn3t'
# print("Downloading test and valid")
# mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy", "Labels-v2.json"], split=["challenge"]) #, "1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"
