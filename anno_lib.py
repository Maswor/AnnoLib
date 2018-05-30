""" Internal Libs for manipulating annotations"""

import csv
import argparse
import sys
from abc import ABCMeta, abstractmethod
from pathlib import Path, PureWindowsPath
import json
import pathlib
from typing import Dict, List, Tuple, TypeVar
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree

from os.path import relpath
import codecs

BOX_TYPE = Tuple[int, int, int, int, str]


def add_shape(bndbox, label):
    """ each line adds (x1, y1, x2, y2, label) """
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    return (xmin, ymin, xmax, ymax, label)


class AnnoParser(metaclass=ABCMeta):
    """Abstract class for xml parser and matlab file parser"""

    def __init__(self, extension, encoding):
        self.xml_ext = extension
        self.encode_method = encoding
        # self.shapes = [{'bbox' =[(x1, y1, x2, y2, 'lable')], 'img_path' = ''}]
        self.shapes = []
        super().__init__()

    @staticmethod
    def create_parser(extension='.xml', encoding='utf-8'):
        """ Delegate parser base on extension """
        if extension == '.xml':
            return XmlParser(extension, encoding)
        elif extension == '.json':
            return MatlabParser(extension, encoding)
        else:
            raise ValueError('Unsupported file type')

    @abstractmethod
    def parse_anno(self):
        """
        abstract method, will be implemented diffirently by children
        parse annotation either form .xml file or matlab json file
        """
        pass

    @property
    def anno_data(self):
        """
        smart getter, just call my_parser.Anno_Data, it will check
        if ParseAnno has run yet and return annotation data.
        self.shapes = [{'bbox' =[(x1, y1, x2, y2, 'lable')], 'img_path' = ''}]

        """
        if self.shapes:
            return self.shapes
        self.parse_anno()
        return self.shapes


class MatlabParser(AnnoParser):
    """
    Parser for annotations made with Matlab
    First convert .mat to .json using
    1st: load('nodboxes.mat')
    2nd: json = jsonencode(Test1)
    3rd: fid = fopen('result.json', 'wt')
    4th: fprintf(fid, '%s', json)
    5th: fclose(fid)
    """

    def __init__(self, extension='.json', encoding='utf-8'):
        """ x, y, width, height """
        self._anno_path = None
        self._imgs_folder = None
        super().__init__(extension, encoding)

    @property
    def imgs_folder(self):
        """ return folder which has images"""
        return self._imgs_folder

    @imgs_folder.setter
    def imgs_folder(self, path):
        self._imgs_folder = Path(path)

    @property
    def anno_path(self):
        """ path to the annotation.json file """
        return self._anno_path

    @anno_path.setter
    def anno_path(self, path):
        self._anno_path = Path(path)

    def fix_img_path(self, img_path):
        """ Fix image location according to provided folder """
        path = PureWindowsPath(img_path)
        my_path = self._imgs_folder / path.name
        return my_path.as_posix()

    def parse_anno(self):
        with open(self._anno_path.as_posix()) as data_file:
            data = json.load(data_file, encoding=self.encode_method)
        for (i, item) in enumerate(data):
            img_path = item['imageFilename']
            img_path = self.fix_img_path(img_path)
            self.shapes.insert(i, {'img_path': img_path})
            self.shapes[i]['bbox'] = []
            for box in item['Nodule']:
                x_1, y_1, _w, _h = box
                # Python count from 0 and slide omit last elem
                self.shapes[i]['bbox'].append((x_1 - 1, y_1 - 1, x_1 + _w,
                                               y_1 + _h, '1'))


class XmlParser(AnnoParser):
    """ xml parser for standard VOC 2007 format """

    def __init__(self, extension='.xml', encoding='utf-8'):
        self.verified = False
        self._folder_path = None
        self.file_paths = None
        super().__init__(extension, encoding)

    @property
    def folder_path(self):
        """ return folder path, note: we use indirection here """
        return self._folder_path

    @folder_path.setter
    def folder_path(self, path):
        """ set path to folder with .xml file """
        self._folder_path = Path(path)
        self.file_paths = sorted(list(self._folder_path.glob('*.xml')))

    def img_abs_path(self, img_rel_path):
        """ return absolute path of image, for traing """
        temp_path = PureWindowsPath(img_rel_path)
        img_rel_path = temp_path.as_posix()
        img_path = self._folder_path / img_rel_path

        return str(img_path.resolve())

    def parse_anno(self):
        for (i, filepath) in enumerate(self.file_paths):
            assert filepath.suffix == self.xml_ext, "unsupported file format"
            parser = etree.XMLParser(encoding=self.encode_method)
            xmltree = ElementTree.parse(
                filepath.as_posix(), parser=parser).getroot()

            img_rel_path = xmltree.find('path').text
            img_path = self.img_abs_path(img_rel_path)
            try:
                verified = xmltree.attrib['verified']
                self.verified = bool(verified)
            except KeyError:
                self.verified = False
            self.shapes.insert(i, {'img_path': img_path})
            self.shapes[i]['bbox'] = []

            for object_iter in xmltree.findall('object'):
                bndbox = object_iter.find('bndbox')
                label = object_iter.find('name').text
                # Assume dificulty False
                my_box = add_shape(bndbox, label)
                self.shapes[i]['bbox'].append(my_box)


class AnnoWriter(metaclass=ABCMeta):
    """ Interface for Annotation Writter classes """

    def __init__(self, parser_obj: AnnoParser, binary: bool = False) -> None:
        self.data = parser_obj.anno_data
        # List[Dict['bbox' =[(x1, y1, x2, y2, 'lable')], 'img_path' = '']]
        self.binary = binary

    @staticmethod
    def create_writer(parser_obj, binary, anno_format):
        """ Delegate parser base on extension """
        if anno_format == 'csv':
            return CSVWriter(parser_obj, binary)
        elif anno_format == 'xml':
            return XMLWriter(parser_obj, binary)
        else:
            raise ValueError('Unsupported file type')

    @abstractmethod
    def write(self):
        """ Writing down the annotations (either to file or folder) """
        pass

    def binary_transform(self, box):
        """ Transform box with label > 1 into label = 1 """
        x_1, y_1, x_2, y_2, label = box
        if self.binary:
            if int(label) >= 1:
                label = '1'
        return (x_1, y_1, x_2, y_2, label)


class CSVWriter(AnnoWriter):
    """ Implementation for writting Retinanet CSV """

    def __init__(self, parser_obj, binary=False):
        super().__init__(parser_obj, binary=False)
        self.filename = None

    def set_output_file(self, file_name):
        """ Set file to store the annotations """
        self.filename = file_name

    def write(self):
        """ Implementation for writting annos """
        with open(self.filename, 'w', newline='') as csvfile:
            anno_writer = csv.writer(
                csvfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            for img_file in self.data:
                for box in img_file['bbox']:
                    box = self.binary_transform(box)
                    if box[4] == '0':
                        continue
                    anno_writer.writerow((img_file['img_path'],) + box)


class XMLWriter(AnnoWriter):
    """ Implementation for writting VOC .xml to destinated folder """

    def __init__(self, parser_obj: AnnoParser, binary: bool) -> None:
        super().__init__(parser_obj, binary)
        self.out_folder = None  # type: pathlib.Path
        self.img_vs_boxes = self._transform_data_to_dict()

    def set_output_folder(self, out_folder: pathlib.Path) -> None:
        """ Set file to store the annotations """
        self.out_folder = out_folder

    def _transform_data_to_dict(self) -> Dict[str, List[BOX_TYPE]]:
        """ Transform data to dict of anno according to img name """
        img_vs_boxes = {}  # type: Dict[str, List[BOX_TYPE]]
        for img_file in self.data:
            for box in img_file['bbox']:
                box = self.binary_transform(box)
                if box[4] == '0':
                    continue
                if img_file['img_path'] not in img_vs_boxes:
                    img_vs_boxes[img_file['img_path']] = []
                img_vs_boxes[img_file['img_path']].append(box)
        return img_vs_boxes

    def write(self):
        """ Writing .xml file to folder """


class PascalVocWriter:

    def __init__(self,
                 foldername,
                 filename,
                 imgSize,
                 databaseSrc='Unknown',
                 xmlFile=None,
                 localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False
        self.xmlFile = xmlFile

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(
            root, pretty_print=True, encoding='utf-8').replace(
                "  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            existed_path = pathlib.Path(self.xmlFile)
            parent_folder = existed_path.parent
            relativePath = relpath(self.localImgPath, str(parent_folder))
            localImgPath = SubElement(top, 'path')
            localImgPath.text = relativePath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object['ymax']) == int(self.imgSize[0]) or (int(
                    each_object['ymin']) == 1):
                truncated.text = "1"  # max == height or min
            elif (int(each_object['xmax']) == int(self.imgSize[1])) or (int(
                    each_object['xmin']) == 1):
                truncated.text = "1"  # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(bool(each_object['difficult']) & 1)
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + '.xml', 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


def parse_args(args):
    """ Parsing a provided argument """
    parser = argparse.ArgumentParser(
        description="Simple parser script for infering Keras-Retinanet on image"
    )
    parser.add_argument(
        'output_file', default='anno.csv', help="Location for csv file")
    parser.add_argument('input_folder', help="Folder which has xml files")
    parser.add_argument(
        '--binary', help="Binary annotation", action='store_true')
    return parser.parse_args(args)


def main(args=None):
    """ Main program, as a function to avoid setting up grobal variables """
    args = parse_args(args)
    my_parser = AnnoParser.create_parser('.xml', 'utf-8')
    my_parser.folder_path = args.input_folder
    my_data = my_parser.anno_data
    my_writer = AnnoWriter.create_writer(
        my_parser, binary=args.binary, anno_format='xml')
    my_writer.set_output_folder('./test')
    my_writer.write()
    total_imgs = 0
    total_boxes = 0
    boxes_indices = [0 for _ in range(10)]
    for img_file in my_data:
        total_imgs += 1
        total_boxes += len(img_file['bbox'])
        for box in img_file['bbox']:
            boxes_indices[int(box[4])] += 1
    print("There are {} images with {} boxes".format(total_imgs, total_boxes))
    print("Of which, number for DS score from 0 - 9 are: {}".format(
        boxes_indices))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))