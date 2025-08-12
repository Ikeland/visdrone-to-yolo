# convert_local_visdrone.py
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm


def visdrone2yolo(dir, split, source_name=None):
    """
    将本地 VisDrone 原始标注转换为 YOLO 格式
    参数:
        dir          – 数据集根目录（含 images 和 labels 输出目录）
        split        – 'train' / 'val' / 'test'
        source_name  – 原始目录名，如 'VisDrone2019-DET-train'；如果为 None，则自动用 'VisDrone2019-DET-{split}'
    """
    source_dir = dir / (source_name or f"VisDrone2019-DET-{split}")
    print(source_dir)
    images_out = dir / "images" / split
    labels_out = dir / "labels" / split
    labels_out.mkdir(parents=True, exist_ok=True)

    # 1. 把原始图片移动到 images/{split}
    if not images_out.exists():
        images_out.mkdir(parents=True, exist_ok=True)
        src_img_dir = source_dir / "images"
        for img in src_img_dir.glob("*.jpg"):
            img.rename(images_out / img.name)

    # 2. 逐文件转换标注
    for anno_file in tqdm((source_dir / "annotations").glob("*.txt"), desc=f"Converting {split}"):
        img_file = images_out / anno_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue  # 跳过无对应图片的标注
        w, h = Image.open(img_file).size
        dw, dh = 1.0 / w, 1.0 / h

        lines = []
        with open(anno_file, encoding="utf-8") as f:
            for row in [x.split(",") for x in f.read().strip().splitlines()]:
                if row[4] == "0":          # 跳过 ignored regions
                    continue
                x, y, bw, bh = map(int, row[:4])
                cls_id = int(row[5]) - 1   # VisDrone 从 1 开始，YOLO 从 0 开始
                xc = (x + bw / 2) * dw
                yc = (y + bh / 2) * dh
                bw_norm = bw * dw
                bh_norm = bh * dh
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

        (labels_out / anno_file.name).write_text("".join(lines), encoding="utf-8")

# 指定本地根目录（请改为自己的路径）
root = Path(r"D:\Desktop\数据处理\A022_训练数据\01_VisDrone\Task1 Object Detection in Images")   # 修改为你的实际路径
print(root)
# 3. 逐个 split 转换
for src_folder, split in {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}.items():
    visdrone2yolo(root, split, src_folder)
    # 若不再需要原始目录可取消下一行注释
    #shutil.rmtree(root / src_folder)