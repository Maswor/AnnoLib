""" Retrieving annotation data from server """
import json
from contextlib import AbstractContextManager
from math import ceil
from pathlib import Path
from os import path
from typing import List, NamedTuple

from mysql import connector

from annolib import AnnoParser, CSVTrainTestWriter, CSVWriter


class BOX_TYPE(NamedTuple):
    x_1: int
    y_1: int
    x_2: int
    y_2: int
    label: str


class AnnosPerImage(NamedTuple):
    bbox: List[BOX_TYPE]
    img_path: str


AnnoDatabase = List[AnnosPerImage]


class DatabaseManager(AbstractContextManager):
    """ Class for managing database """

    def __init__(self, user: str, password: str, host: str,
                 database: str) -> None:
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.release_resource: bool = True
        self.cnx: connector.connection.MySQLConnection

    def __enter__(self) -> connector.cursor.MySQLCursor:
        print("__Aquiring Database resource__")
        try:
            self.cnx = connector.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                database=self.database)
        except connector.errors.Error:
            self.release_resource = False
        else:
            return self.cnx.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.release_resource:
            self.cnx.close()
            print("__Release Database resource__")
        else:
            print("__Failed to aquire resource__")


def convert_to_twopoint_bboxes(weird_boxes: List[List[int]],
                               ds_score: str) -> List[BOX_TYPE]:
    """ convert a list of x, y, width, height (can be negative) """
    return_data: List[BOX_TYPE] = []
    for weird_box in weird_boxes:
        start_x, start_y, width, height = weird_box
        end_x = start_x + width
        end_y = start_y + height
        x_1 = ceil(min(start_x, end_x))
        y_1 = ceil(min(start_y, end_y))
        x_2 = ceil(max(start_x, end_x))
        y_2 = ceil(max(start_y, end_y))
        if (x_2 - x_1) * (y_2 - y_1) > 100:
            # TODO debug
            min_box = min(abs(width), abs(height))
            compress_ratio = 1200 / 3456
            compress_min_box = min_box * compress_ratio
            if compress_min_box < 32:
                print("The size of box {} after compress is {}. Bad".format(
                    min_box, compress_min_box))
            # TODO enddebug
            return_data.append(BOX_TYPE(x_1, y_1, x_2, y_2, ds_score))
    return return_data


def transform_json_boxes(json_boxes: bytes) -> List[BOX_TYPE]:
    """ transform json strem to boxes, a box can be empty !!! """
    try:
        decoded_bboxes = json.loads(json_boxes)
    except json.decoder.JSONDecodeError:
        print("Error input detected, skipping problematic line")
        return []
    return_data: List[BOX_TYPE] = []
    for ds_score, weird_boxes in decoded_bboxes.items():
        normal_boxes = convert_to_twopoint_bboxes(weird_boxes, ds_score)
        return_data.extend(normal_boxes)
    return return_data


class MySQLParser(AnnoParser):
    """ Parser for MySQL database """

    def __init__(self, encoding: str, source_folder: str,
                 new_csv_folder: str) -> None:
        """ Constructor
        source_folder: parent folder which store images file
        new_csv_folder: folder which will store the csv file
        Need for new Keras - Retinanet
        """
        super().__init__(encoding)
        self.source_folder = Path(source_folder)
        self.new_csv_foler = Path(new_csv_folder)

    #TODO original path "/Hindsfarm/SDS" we have to remove first slash "Hindsfarm/SDS
    def transform_path(self, relative_path: str) -> str:
        """ Append parent folder to relative path """
        full_path = self.source_folder / relative_path[1:]
        return path.relpath(full_path, self.new_csv_foler)

    def parse_anno(self) -> AnnoDatabase:
        return_data: AnnoDatabase = []
        with DatabaseManager("root", "Icui4cuss", "localhost",
                             "soybean_tagger") as my_cusor:
            my_cusor.execute(
                """select `MarkedData`.path, `Images`.path from `MarkedData` inner 
                join `Images` on `MarkedData`.image_id = `Images`.image_id where 
                `MarkedData`.author = 'Randi' """)
            for json_boxes, relative_path in my_cusor:
                # print("boxes: {} and relative_path: {}".format(
                #     json_boxes, relative_path))
                bboxes = transform_json_boxes(json_boxes)
                img_path = self.transform_path(relative_path)
                if bboxes:
                    return_data.append(AnnosPerImage(bboxes, img_path))
        return return_data


def main() -> None:
    """ Main function """
    parser = MySQLParser("utf-8",
                         "~/Desktop/PointCloudISU/WebApp/www/html/sds_images",
                         "~/Desktop/PointCloudISU/sdsgwas/src/traing_data")
    # writer = CSVWriter(parser, True)
    # writer.set_output_file("total.csv")
    # writer.write()
    writer = CSVTrainTestWriter(parser, 0.8, False)
    writer.set_output_file("train.csv", "test.csv")
    writer.write()


if __name__ == "__main__":
    main()
