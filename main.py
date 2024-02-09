import os
import argparse
from detector import run_model2annotations
from cbz import unzip_cbz
import shutil
def process_zip_directory(directory, file, annotation_dir, panel_dir):
    run_model2annotations(directory, f'{annotation_dir}/{os.path.basename(file)}', save_json=True)
    shutil.rmtree(directory)

def process_directory(file_path, annotation_dir, panel_dir):
    dir = os.path.basename(os.path.dirname(file_path))
    run_model2annotations(file_path, f'{annotation_dir}/{dir}', save_json=True)

def main(input_dir, output_dir, annotation_dir, panel_dir):
    for file in os.listdir(input_dir):
        if file.endswith(('.cbz', '.zip', '.rar')):
            file_name = os.path.splitext(file)[0]
            output_subdir = os.path.join(output_dir, file_name)
            unzip_cbz(os.path.join(input_dir, file), output_subdir)
            process_zip_directory(output_subdir, file, annotation_dir, panel_dir)
        elif file.endswith(('.jpg', '.png', '.jpeg')):
            process_directory(input_dir, annotation_dir, panel_dir)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble Detector CLI")
    parser.add_argument("--input_dir", required=True, help="Directory containing the input files (cbz, zip, rar, jpg, png, jpeg)")
    parser.add_argument("--output_dir", default="./manga_pages", help="Directory to save the extracted manga pages")
    parser.add_argument("--annotation_dir", default="./output/annotations", help="Directory to save the speech bubble annotations")
    parser.add_argument("--panel_dir", default="./output/panels", help="Directory to save the panel images")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.annotation_dir, args.panel_dir)
