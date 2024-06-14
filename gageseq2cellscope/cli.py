import argparse
from core import generate_hic_file

def main():
    parser = argparse.ArgumentParser(description="Generate scHiC HDF5 files from Gage-Seq")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['create', 'append', 'print', 'check'], help='Mode for the operation')
    parser.add_argument('-t', '--types', nargs='+', help='Types of operation for append')

    args = parser.parse_args()

    if args.mode == 'append' and args.types is None:
        parser.error("The -t/--types option is required when mode is 'append'")

    if args.mode != 'append' and args.types is not None:
        parser.error("The -t/--types option is only available when mode is 'append'")

    generate_hic_file(args.config, args.mode, args.types)

if __name__ == "__main__":
    main()