#!/usr/bin/env python3
import argparse, json, shutil, pathlib, csv

def copy_tree(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='original TinyImageNet root')
    p.add_argument('--dst', required=True, help='output root for reduced set')
    p.add_argument('--keep-list', required=True, help='JSON file with synsets to keep')
    args = p.parse_args()

    src = pathlib.Path(args.src).expanduser().resolve()
    dst = pathlib.Path(args.dst).expanduser().resolve()
    keep = set(json.loads(pathlib.Path(args.keep_list).read_text()))

    # 1. Copy training images
    for syn in keep:
        src_img_dir = src / 'train' / syn / 'images'
        dst_img_dir = dst / 'train' / syn / 'images'
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for img in src_img_dir.iterdir():
            copy_tree(img, dst_img_dir / img.name)

    # 2. Filter validation set
    val_src_img = src / 'val' / 'images'
    val_src_ann = src / 'val' / 'val_annotations.txt'
    dst_val_root = dst / 'val'
    dst_val_root.mkdir(parents=True, exist_ok=True)
    dst_ann = dst_val_root / 'val_annotations.txt'

    with val_src_ann.open() as f_in, dst_ann.open('w') as f_out:
        writer = csv.writer(f_out, delimiter='\t', lineterminator='\n')
        for row in csv.reader(f_in, delimiter='\t'):
            img, syn = row[0], row[1]
            if syn in keep:
                src_path = val_src_img / img
                dst_path = dst_val_root / syn / 'images' / img
                copy_tree(src_path, dst_path)
                writer.writerow(row)

    print(f'Done. Reduced dataset saved to {dst}')

if __name__ == '__main__':
    main()
