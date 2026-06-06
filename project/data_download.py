from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse


URL_PATTERN = re.compile(r"https?://[^\s)>\"']+")
DATA_URL_PATTERN = re.compile(r"^\s*(?:HARTH_)?DATA_URL\s*[:=]\s*(https?://\S+)\s*$", re.IGNORECASE)


def find_data_url(readme_path: Path) -> str | None:
    if not readme_path.exists():
        return None

    text = readme_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = DATA_URL_PATTERN.search(line)
        if match and is_real_data_url(match.group(1)):
            return cleanup_url(match.group(1))

    for url in URL_PATTERN.findall(text):
        cleaned = cleanup_url(url)
        if is_real_data_url(cleaned) and ("drive.google.com" in cleaned or cleaned.endswith((".zip", ".csv"))):
            return cleaned
    return None


def cleanup_url(url: str) -> str:
    return url.strip().rstrip(".,;")


def is_real_data_url(url: str) -> bool:
    placeholders = ("FILE_ID", "YOUR_", "<", ">")
    return url.startswith(("http://", "https://")) and not any(item in url for item in placeholders)


def has_csv_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.csv"))


def find_csv_dir(root: Path) -> Path | None:
    if has_csv_files(root):
        return root

    candidates: list[tuple[int, Path]] = []
    if root.exists():
        for path in root.rglob("*"):
            if path.is_dir():
                count = len(list(path.glob("*.csv")))
                if count:
                    candidates.append((count, path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def download_data_from_url(url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    download_dir = target_dir.parent / "_download_cache"
    download_dir.mkdir(parents=True, exist_ok=True)

    if "drive.google.com" in url:
        return download_from_google_drive(url, target_dir, download_dir)

    archive_path = download_regular_url(url, download_dir)
    return unpack_downloaded_file(archive_path, target_dir)


def download_from_google_drive(url: str, target_dir: Path, download_dir: Path) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Link dữ liệu là Google Drive nhưng thiếu thư viện gdown. "
            "Chạy: pip install -r requirements.txt"
        ) from exc

    if is_google_drive_folder(url):
        print(f"Downloading HARTH folder from Google Drive to: {target_dir}")
        gdown.download_folder(url=url, output=str(target_dir), quiet=False, use_cookies=False)
        csv_dir = find_csv_dir(target_dir)
        if csv_dir is None:
            raise FileNotFoundError("Đã tải Google Drive folder nhưng không tìm thấy file CSV.")
        return csv_dir

    output_path = download_dir / "harth_data_download"
    print(f"Downloading HARTH archive from Google Drive to: {output_path}")
    downloaded = gdown.download(url=url, output=str(output_path), quiet=False, fuzzy=True, use_cookies=False)
    if downloaded is None:
        raise RuntimeError("Không tải được dữ liệu từ Google Drive. Kiểm tra quyền chia sẻ link.")
    return unpack_downloaded_file(Path(downloaded), target_dir)


def is_google_drive_folder(url: str) -> bool:
    parsed = urlparse(url)
    return "drive.google.com" in parsed.netloc and ("/folders/" in parsed.path or "folders" in parsed.path)


def download_regular_url(url: str, download_dir: Path) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện requests để tải dữ liệu. Chạy: pip install -r requirements.txt"
        ) from exc

    filename = Path(urlparse(url).path).name or "harth_data_download"
    output_path = download_dir / filename
    print(f"Downloading HARTH data to: {output_path}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
    return output_path


def unpack_downloaded_file(downloaded_path: Path, target_dir: Path) -> Path:
    if zipfile.is_zipfile(downloaded_path):
        print(f"Extracting HARTH data to: {target_dir}")
        safe_extract_zip(downloaded_path, target_dir)
        csv_dir = find_csv_dir(target_dir)
        if csv_dir is None:
            raise FileNotFoundError("Đã giải nén dữ liệu nhưng không tìm thấy file CSV.")
        return csv_dir

    if downloaded_path.suffix.lower() == ".csv":
        target_path = target_dir / downloaded_path.name
        if downloaded_path.resolve() != target_path.resolve():
            shutil.copy2(downloaded_path, target_path)
        return target_dir

    raise ValueError(
        "File dữ liệu tải về không phải .zip hoặc .csv. "
        "Khuyến nghị upload Source/Task2/data dưới dạng data.zip lên Google Drive."
    )


def safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (target_dir / member.filename).resolve()
            if target_root not in destination.parents and destination != target_root:
                raise RuntimeError(f"Zip chứa đường dẫn không an toàn: {member.filename}")
        archive.extractall(target_dir)
