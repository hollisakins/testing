import argparse
from . import config
import sys, os


def main_cli():
    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=config.cols)
    parser = argparse.ArgumentParser(
                        prog='brisket',
                        description='Command line interface to BRISKET',
                        formatter_class=formatter)
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode")

    parser_cloudy = subparsers.add_parser("run_cloudy", help="Run cloudy on a given grid")
    parser_cloudy.add_argument('grid', help='Grid file name', type=str)


    parser_grids = subparsers.add_parser("grids", help="Manage grids")
    parser_grids.add_argument('--list', help='List all grids', action='store_true')
    parser_grids.add_argument('--download_all', help='Download all grids from bucket', action='store_true')
    # parser_grids.add_argument('--download', help='Download a specific grid', type=str)
    parser_grids.add_argument('-b', '--bucket', help='S3 bucket name', default='brisket-data')


    args = parser.parse_args()

    ############################################################################################################################################################
    if args.mode == 'grids':
        from .data.grid_manager import GridManager
        gm = GridManager(bucket=args.bucket)

        if args.list: 
            objs = gm.list_objects()
            for obj in objs:
                grid_file_name = obj['Key']
                if grid_file_name.endswith('.fits') or grid_file_name.endswith('.hdf5'):
                    if os.path.exists(os.path.join(config.grid_dir, grid_file_name)):
                        print(f":file_folder: {obj['Key']}: {gm.format_size(obj['Size'])} :white_check_mark:")
                    else:
                        print(f":file_folder: [red]{obj['Key']}[/red]: {gm.format_size(obj['Size'])} :cross_mark:")

        if args.download_all:
            total_size = 0
            names_to_download = []
            print('Grid files to download:')
            objs = gm.list_objects()
            for obj in objs:
                grid_file_name = obj['Key']
                if grid_file_name.endswith('.fits') or grid_file_name.endswith('.hdf5'):
                    if os.path.exists(os.path.join(config.grid_dir, grid_file_name)):
                        print(f":file_folder: {grid_file_name}: {gm.format_size(obj['Size'])} :white_check_mark:")
                    else:
                        print(f":file_folder: [red]{grid_file_name}[/red]: {gm.format_size(obj['Size'])} :cross_mark:")
                        names_to_download.append(grid_file_name)
                        total_size += obj['Size']

            whether_continue = Confirm.ask(f"Would you like to download all [[red]{gm.format_size(total_size)}[/red]] from [blue]s3://{args.bucket}[/blue]?", default=True)
            if not whether_continue: sys.exit(1)
            for grid_file_name in names_to_download:
                gm.download_file(grid_file_name, os.path.join(config.grid_dir, grid_file_name))


    ############################################################################################################################################################
    if args.mode == 'run_cloudy':
        print(args.grid)