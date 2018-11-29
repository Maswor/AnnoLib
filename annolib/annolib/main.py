import argparse
import pathlib
import sys
from typing import List

from annolib.annolib import (AnnoDatabase, AnnoParser, AnnoWriter, CSVParser,
                             CSVWriter, MatlabParser, XmlParser, XMLWriter)


def print_info(anno_data: AnnoDatabase) -> None:
    """ Printing boxes information """
    total_imgs = 0
    total_boxes = 0
    boxes_indices = [0 for _ in range(10)]
    for img_file in anno_data:
        total_imgs += 1
        total_boxes += len(img_file.bbox)
        for box in img_file.bbox:
            boxes_indices[int(box[4])] += 1
    print("There are {} images with {} boxes".format(total_imgs, total_boxes))
    print("Of which, number for DS score from 0 - 9 are: {}".format(
        boxes_indices))


def select_writer(arg_output: str, parser: AnnoParser,
                  binary: bool) -> AnnoWriter:
    """ Writer selection depending on arg_output """
    m_output = pathlib.Path(arg_output)
    my_writer: AnnoWriter
    if m_output.is_dir():
        my_writer = XMLWriter(parser, binary)
        my_writer.set_output_folder(arg_output)
    elif m_output.suffix == '.csv':
        my_writer = CSVWriter(parser, binary)
        my_writer.set_output_file(arg_output)
    else:
        raise ValueError(
            "Output path doesn't exist or unsupported writing method")

    return my_writer


def select_parser(arg_input: str) -> AnnoParser:
    """ Parser selection depending on arg_input """
    m_input = pathlib.Path(arg_input)
    my_parser: AnnoParser
    if m_input.is_dir():
        my_parser = XmlParser("utf-8")
        my_parser.folder_path = m_input
    elif m_input.suffix == '.csv':
        my_parser = CSVParser('utf-8')
        my_parser.set_anno_file(arg_input)
    elif m_input.suffix == '.json':
        my_parser = MatlabParser('utf-8')
        my_parser.anno_path = m_input
    else:
        raise ValueError(
            "Input path doesn't exist or unsupported Reading method")

    return my_parser


def parse_args(args: List[str]) -> argparse.Namespace:
    """ Parsing a provided argument """
    parser = argparse.ArgumentParser(
        description="Simple parser script for infering Keras-Retinanet on image"
    )
    parser.add_argument(
        'input',
        help="Input annotation, either folder (of .xml) or file (.csv, .json)")
    parser.add_argument(
        'output',
        help="Output annotation, either file (of .csv) or folder (.xml, .json)"
    )
    parser.add_argument(
        '--binary', help="Binary annotation", action='store_true')
    return parser.parse_args(args)


def main() -> None:
    """ Main program, as a function to avoid setting up grobal variables """
    p_args = parse_args(sys.argv[1:])
    my_parser = select_parser(p_args.input)
    my_writer = select_writer(p_args.output, my_parser, p_args.binary)
    my_writer.write()
