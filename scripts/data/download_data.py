import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "config" / "datasets.json"
METADATA_DIR = ROOT / "data" / "metadata"
RAW_DIR = ROOT / "data" / "raw"

CHUNK_SIZE = 1024 * 1024


def load_datasets():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_request(url, method=None):
    return urllib.request.Request(
        url,
        method=method,
        headers={
            "User-Agent": "CMPS364-data-acquisition/1.0",
        },
    )


def fetch_json(url):
    request = make_request(url)

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    except urllib.error.HTTPError as error:
        print(
            f"HTTP error {error.code}: {error.reason}",
            file=sys.stderr,
        )
        sys.exit(1)

    except urllib.error.URLError as error:
        print(
            f"Connection error: {error.reason}",
            file=sys.stderr,
        )
        sys.exit(1)


def format_size(size_bytes):
    if size_bytes is None:
        return "unknown"

    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} B"


def calculate_hash(path, algorithm):
    digest = hashlib.new(algorithm)

    with path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json_file(path):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def get_dataverse_metadata(dataset):
    query = urllib.parse.urlencode(
        {
            "persistentId": dataset["persistent_id"],
            "excludeFiles": "false",
        }
    )

    url = (
        f'{dataset["server"]}'
        f'/api/datasets/:persistentId/'
        f'versions/:latest-published?{query}'
    )

    return fetch_json(url)


def dataverse_metadata_path(dataset_key):
    return (
        METADATA_DIR
        / f"{dataset_key}_dataverse.json"
    )


def save_dataverse_metadata(dataset_key, payload):
    path = dataverse_metadata_path(
        dataset_key
    )

    save_json_file(
        path,
        payload,
    )

    return path


def get_dataverse_filename(file_info):
    data_file = file_info.get(
        "dataFile",
        {},
    )

    return file_info.get(
        "label",
        data_file.get(
            "filename",
            "<unknown>",
        ),
    )


def get_dataverse_md5(file_info):
    data_file = file_info.get(
        "dataFile",
        {},
    )

    checksum = data_file.get(
        "checksum"
    )

    if (
        checksum
        and checksum.get(
            "type",
            "",
        ).upper()
        == "MD5"
    ):
        return checksum.get("value")

    return data_file.get("md5")


def list_dataverse_dataset(
    dataset_key,
    dataset,
):
    payload = get_dataverse_metadata(
        dataset
    )

    if payload.get("status") != "OK":
        print(
            "Dataverse returned an unexpected response.",
            file=sys.stderr,
        )
        sys.exit(1)

    metadata_path = save_dataverse_metadata(
        dataset_key,
        payload,
    )

    data = payload["data"]

    files = data.get(
        "files",
        [],
    )

    major = data.get(
        "versionNumber",
        "?",
    )

    minor = data.get(
        "versionMinorNumber",
        "?",
    )

    total_size = sum(
        file_info.get(
            "dataFile",
            {},
        ).get(
            "filesize",
            0,
        )
        for file_info in files
    )

    print(
        f"Dataset: {dataset['name']}"
    )

    print(
        f"Provider: {dataset['provider']}"
    )

    print(
        f"DOI: {dataset['doi']}"
    )

    print()

    print(
        f"Published version: {major}.{minor}"
    )

    print(
        f"Files: {len(files)}"
    )

    print(
        f"Total size: {format_size(total_size)}"
    )

    print(
        "Metadata saved to: "
        f"{metadata_path.relative_to(ROOT)}"
    )

    print()
    print("File inventory:")
    print("-" * 90)

    for index, file_info in enumerate(
        files,
        start=1,
    ):
        data_file = file_info.get(
            "dataFile",
            {},
        )

        filename = get_dataverse_filename(
            file_info
        )

        file_id = data_file.get(
            "id",
            "?",
        )

        filesize = data_file.get(
            "filesize",
            0,
        )

        restricted = file_info.get(
            "restricted",
            False,
        )

        md5 = (
            get_dataverse_md5(
                file_info
            )
            or "-"
        )

        print(
            f"{index:>2}. "
            f"id={file_id:<8} "
            f"size={format_size(filesize):<12} "
            f"restricted={str(restricted):<5} "
            f"{filename}"
        )

        print(
            f"    checksum=MD5:{md5}"
        )


def find_dataverse_file(
    payload,
    filename,
):
    for file_info in payload[
        "data"
    ].get(
        "files",
        [],
    ):
        if (
            get_dataverse_filename(
                file_info
            )
            == filename
        ):
            return file_info

    return None


def download_dataverse_file(
    dataset_key,
    dataset,
    filename,
):
    payload = get_dataverse_metadata(
        dataset
    )

    if payload.get("status") != "OK":
        print(
            "Dataverse returned an unexpected response.",
            file=sys.stderr,
        )
        sys.exit(1)

    save_dataverse_metadata(
        dataset_key,
        payload,
    )

    file_info = find_dataverse_file(
        payload,
        filename,
    )

    if file_info is None:
        print(
            f"File '{filename}' was not found.",
            file=sys.stderr,
        )

        print(
            "Available files:",
            file=sys.stderr,
        )

        for item in payload[
            "data"
        ].get(
            "files",
            [],
        ):
            print(
                f"  - "
                f"{get_dataverse_filename(item)}",
                file=sys.stderr,
            )

        sys.exit(1)

    if file_info.get(
        "restricted",
        False,
    ):
        print(
            f"File '{filename}' is restricted.",
            file=sys.stderr,
        )
        sys.exit(1)

    data_file = file_info[
        "dataFile"
    ]

    file_id = data_file["id"]

    expected_md5 = get_dataverse_md5(
        file_info
    )

    destination_dir = (
        RAW_DIR
        / dataset_key
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        destination_dir
        / filename
    )

    if (
        destination.exists()
        and expected_md5
    ):
        current_md5 = calculate_hash(
            destination,
            "md5",
        )

        if (
            current_md5.lower()
            == expected_md5.lower()
        ):
            print(
                "Already downloaded and verified: "
                f"{destination.relative_to(ROOT)}"
            )
            return

        print(
            "Existing file failed checksum verification. "
            "Re-downloading."
        )

    url = (
        f'{dataset["server"]}'
        f'/api/access/datafile/'
        f'{file_id}'
    )

    print(
        f"Downloading: {filename}"
    )

    print(
        f"Source: {url}"
    )

    print(
        "Destination: "
        f"{destination.relative_to(ROOT)}"
    )

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    request = make_request(
        url
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            total = response.headers.get(
                "Content-Length"
            )

            total = (
                int(total)
                if total
                else None
            )

            downloaded = 0
            next_progress = 10

            with temporary.open(
                "wb"
            ) as output:

                while True:
                    chunk = response.read(
                        CHUNK_SIZE
                    )

                    if not chunk:
                        break

                    output.write(chunk)

                    downloaded += len(
                        chunk
                    )

                    if total:
                        percent = (
                            downloaded
                            * 100
                            / total
                        )

                        if (
                            percent
                            >= next_progress
                        ):
                            print(
                                f"Progress: "
                                f"{percent:.0f}% "
                                f"({format_size(downloaded)} "
                                f"/ {format_size(total)})"
                            )

                            next_progress += 10

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as error:

        print(
            f"Download failed: {error}",
            file=sys.stderr,
        )

        sys.exit(1)

    if expected_md5:
        actual_md5 = calculate_hash(
            temporary,
            "md5",
        )

        if (
            actual_md5.lower()
            != expected_md5.lower()
        ):
            temporary.unlink(
                missing_ok=True
            )

            print(
                "Checksum verification failed.",
                file=sys.stderr,
            )

            print(
                f"Expected: {expected_md5}",
                file=sys.stderr,
            )

            print(
                f"Actual:   {actual_md5}",
                file=sys.stderr,
            )

            sys.exit(1)

        print(
            f"MD5 verified: {actual_md5}"
        )

    temporary.replace(
        destination
    )

    print(
        "Saved: "
        f"{destination.relative_to(ROOT)}"
    )


def direct_metadata_path(
    dataset_key,
):
    return (
        METADATA_DIR
        / f"{dataset_key}_files.json"
    )


def direct_manifest_path(
    dataset_key,
):
    return (
        METADATA_DIR
        / f"{dataset_key}_downloads.json"
    )


def probe_remote_file(url):
    request = make_request(
        url,
        method="HEAD",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            content_length = (
                response.headers.get(
                    "Content-Length"
                )
            )

            return {
                "final_url":
                    response.geturl(),

                "size_bytes":
                    int(content_length)
                    if content_length
                    else None,

                "content_type":
                    response.headers.get(
                        "Content-Type"
                    ),

                "last_modified":
                    response.headers.get(
                        "Last-Modified"
                    ),

                "etag":
                    response.headers.get(
                        "ETag"
                    ),
            }

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
    ):
        return {
            "final_url":
                url,

            "size_bytes":
                None,

            "content_type":
                None,

            "last_modified":
                None,

            "etag":
                None,
        }


def list_direct_dataset(
    dataset_key,
    dataset,
):
    discovered = {}

    print(
        f"Dataset: {dataset['name']}"
    )

    print(
        f"Provider: {dataset['provider']}"
    )

    print(
        f"Source: {dataset['page_url']}"
    )

    print()
    print("Required file inventory:")
    print("-" * 90)

    for alias, file_info in dataset[
        "files"
    ].items():

        remote = probe_remote_file(
            file_info["url"]
        )

        discovered[alias] = {
            **file_info,
            **remote,
        }

        print(
            f"{alias}:"
        )

        print(
            f"  file: "
            f"{file_info['filename']}"
        )

        print(
            f"  purpose: "
            f"{file_info['description']}"
        )

        print(
            f"  size: "
            f"{format_size(remote['size_bytes'])}"
        )

        print(
            f"  url: "
            f"{file_info['url']}"
        )

        print()

    payload = {
        "dataset":
            dataset["name"],

        "provider":
            dataset["provider"],

        "page_url":
            dataset["page_url"],

        "files":
            discovered,
    }

    path = direct_metadata_path(
        dataset_key
    )

    save_json_file(
        path,
        payload,
    )

    print(
        "Metadata saved to: "
        f"{path.relative_to(ROOT)}"
    )


def download_direct_file(
    dataset_key,
    dataset,
    alias,
):
    files = dataset[
        "files"
    ]

    if alias not in files:
        print(
            f"Unknown file alias '{alias}'.",
            file=sys.stderr,
        )

        print(
            "Available aliases:",
            file=sys.stderr,
        )

        for available in files:
            print(
                f"  - {available}",
                file=sys.stderr,
            )

        sys.exit(1)

    file_info = files[
        alias
    ]

    filename = file_info[
        "filename"
    ]

    url = file_info[
        "url"
    ]

    destination_dir = (
        RAW_DIR
        / dataset_key
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        destination_dir
        / filename
    )

    manifest_path = (
        direct_manifest_path(
            dataset_key
        )
    )

    manifest = load_json_file(
        manifest_path
    )

    manifest.setdefault(
        "dataset",
        dataset["name"],
    )

    manifest.setdefault(
        "files",
        {},
    )

    previous = manifest[
        "files"
    ].get(
        alias
    )

    if (
        destination.exists()
        and previous
        and previous.get(
            "sha256"
        )
    ):
        print(
            "Existing file found. "
            "Verifying SHA-256..."
        )

        actual_sha256 = (
            calculate_hash(
                destination,
                "sha256",
            )
        )

        if (
            actual_sha256
            == previous["sha256"]
        ):
            print(
                "Already downloaded and verified: "
                f"{destination.relative_to(ROOT)}"
            )

            print(
                f"SHA-256: {actual_sha256}"
            )

            return

        print(
            "Existing file does not match "
            "the recorded manifest. "
            "Re-downloading."
        )

    print(
        f"Downloading: {alias}"
    )

    print(
        f"File: {filename}"
    )

    print(
        f"Source: {url}"
    )

    print(
        "Destination: "
        f"{destination.relative_to(ROOT)}"
    )

    request = make_request(
        url
    )

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            content_length = (
                response.headers.get(
                    "Content-Length"
                )
            )

            expected_size = (
                int(content_length)
                if content_length
                else None
            )

            downloaded = 0
            next_progress = 10

            with temporary.open(
                "wb"
            ) as output:

                while True:
                    chunk = response.read(
                        CHUNK_SIZE
                    )

                    if not chunk:
                        break

                    output.write(
                        chunk
                    )

                    digest.update(
                        chunk
                    )

                    downloaded += len(
                        chunk
                    )

                    if expected_size:
                        percent = (
                            downloaded
                            * 100
                            / expected_size
                        )

                        if (
                            percent
                            >= next_progress
                        ):
                            print(
                                f"Progress: "
                                f"{percent:.0f}% "
                                f"({format_size(downloaded)} "
                                f"/ "
                                f"{format_size(expected_size)})"
                            )

                            next_progress += 10

            response_metadata = {
                "content_type":
                    response.headers.get(
                        "Content-Type"
                    ),

                "last_modified":
                    response.headers.get(
                        "Last-Modified"
                    ),

                "etag":
                    response.headers.get(
                        "ETag"
                    ),

                "final_url":
                    response.geturl(),
            }

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as error:

        print(
            f"Download failed: {error}",
            file=sys.stderr,
        )

        sys.exit(1)

    if (
        expected_size is not None
        and downloaded != expected_size
    ):
        temporary.unlink(
            missing_ok=True
        )

        print(
            "Download size verification failed.",
            file=sys.stderr,
        )

        print(
            f"Expected: "
            f"{expected_size} bytes",
            file=sys.stderr,
        )

        print(
            f"Actual:   "
            f"{downloaded} bytes",
            file=sys.stderr,
        )

        sys.exit(1)

    sha256 = digest.hexdigest()

    temporary.replace(
        destination
    )

    manifest[
        "files"
    ][alias] = {
        "filename":
            filename,

        "description":
            file_info[
                "description"
            ],

        "source_url":
            url,

        "size_bytes":
            downloaded,

        "sha256":
            sha256,

        **response_metadata,
    }

    save_json_file(
        manifest_path,
        manifest,
    )

    print(
        "Saved: "
        f"{destination.relative_to(ROOT)}"
    )

    print(
        f"Size: "
        f"{format_size(downloaded)}"
    )

    print(
        f"SHA-256: {sha256}"
    )

    print(
        "Manifest updated: "
        f"{manifest_path.relative_to(ROOT)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible dataset acquisition "
            "for the CMPS 364 eating-behavior project."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset key defined "
            "in config/datasets.json."
        ),
    )

    actions = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    actions.add_argument(
        "--list",
        action="store_true",
        help=(
            "List the files required "
            "from the dataset."
        ),
    )

    actions.add_argument(
        "--file",
        help=(
            "Download one file. "
            "For KU Leuven use the published filename. "
            "For Clemson use a configured alias."
        ),
    )

    args = parser.parse_args()

    datasets = load_datasets()

    if args.dataset not in datasets:
        available = ", ".join(
            sorted(
                datasets.keys()
            )
        )

        parser.error(
            f"Unknown dataset "
            f"'{args.dataset}'. "
            f"Available datasets: "
            f"{available}"
        )

    dataset = datasets[
        args.dataset
    ]

    dataset_type = dataset.get(
        "type"
    )

    if dataset_type == "dataverse":

        if args.list:
            list_dataverse_dataset(
                args.dataset,
                dataset,
            )

        elif args.file:
            download_dataverse_file(
                args.dataset,
                dataset,
                args.file,
            )

        return

    if dataset_type == "direct_files":

        if args.list:
            list_direct_dataset(
                args.dataset,
                dataset,
            )

        elif args.file:
            download_direct_file(
                args.dataset,
                dataset,
                args.file,
            )

        return

    parser.error(
        f"Unsupported dataset type: "
        f"{dataset_type!r}"
    )


if __name__ == "__main__":
    main()