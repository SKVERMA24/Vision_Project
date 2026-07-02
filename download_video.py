import urllib.request
import os
import sys

def download_video(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        def report_hook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 1e2 / total_size
                s = f"\rProgress: {percent:.1f}% ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)"
                sys.stdout.write(s)
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rRead {read_so_far / (1024*1024):.2f} MB")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nFailed to download: {e}")
        # Try fallback URL
        fallback_url = "https://raw.githubusercontent.com/udacity/CarND-Advanced-Lane-Lines/master/project_video.mp4"
        print(f"Trying fallback URL: {fallback_url}")
        try:
            urllib.request.urlretrieve(fallback_url, dest_path, reporthook=report_hook)
            print("\nDownload complete from fallback!")
        except Exception as e2:
            print(f"\nFailed to download from fallback: {e2}")

if __name__ == "__main__":
    url = "https://github.com/aminnour/Lane-Detection/raw/master/project_video.mp4"
    dest = "project_video.mp4"
    download_video(url, dest)
