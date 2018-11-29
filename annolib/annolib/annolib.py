""" Internal Libs for manipulating annotations, Require Python 3"""
import codecs
import csv
import json
import pathlib
import warnings
from abc import ABCMeta, abstractmethod
from os.path import relpath
from typing import Dict, List, NamedTuple, Optional, Tuple
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

import cv2
from lxml import etree


# Original List[Dict['bbox' =[(x1, y1, x2, y2, 'label')], 'img_path' = '']]
class BOX_TYPE(NamedTuple):
    x_1: int
    y_1: int
    x_2: int
    y_2: int
    label: str


class PASCAL_BOX_TYPE(NamedTuple):
    x_1: int
    y_1: int
    x_2: int
    y_2: int
    label: str
    difficulty: bool


ImageAnnoDict = Dict[str, List[BOX_TYPE]]


class AnnosPerImage(NamedTuple):
    bbox: List[BOX_TYPE]
    img_path: str


AnnoDatabase = List[AnnosPerImage]


class AnnoParser(metaclass=ABCMeta):
    """Abstract class for xml parser and matlab file parser"""

    def __init__(self, encoding: str) -> None:
        self.encode_method = encoding
        self.shapes: AnnoDatabase = []
        super().__init__()

    @abstractmethod
    def parse_anno(self) -> None:
        """
        abstract method, will be implemented diffirently by children
        parse annotation either form .xml file or matlab json file
        """
        pass

    @property
    def anno_data(self) -> AnnoDatabase:
        """
        smart getter, just call my_parser.Anno_Data, it will check
        if ParseAnno has run yet and return annotation data.
        self.shapes = [{'bbox' =[(x1, y1, x2, y2, 'label')], 'img_path' = ''}]

        """
        # Empty List valuates to fall
        if self.shapes:
            return self.shapes
        self.parse_anno()
        return self.shapes

    @anno_data.setter
    def anno_data(self, data) -> None:
        self.shapes = data


class CSVParser(AnnoParser):
    """ CSV (i.e. Retinanet) annotations parser """

    def __init__(self, encoding: str) -> None:
        super().__init__(encoding)
        self.anno_file: Optional[pathlib.Path] = None

    def set_anno_file(self, path: str) -> None:
        """ Add annotations for given file """
        self.anno_file = pathlib.Path(path)

    def parse_anno(self) -> None:
        proto_data = self._create_proto_anno()
        for path, custom_anno in proto_data.items():
            img_annos = AnnosPerImage(custom_anno, path)
            self.shapes.append(img_annos)

    def _create_proto_anno(self) -> ImageAnnoDict:
        """ intermediate representation of annotations """
        proto_data: ImageAnnoDict = {}
        assert self.anno_file is not None
        with self.anno_file.open('r', newline='') as csvfile:
            anno_reader = csv.reader(
                csvfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            for anno_line in anno_reader:
                img_path = anno_line[0]
                box = BOX_TYPE(
                    int(float(anno_line[1])), int(float(anno_line[2])),
                    int(float(anno_line[3])), int(float(anno_line[4])),
                    anno_line[5])
                if img_path not in proto_data:
                    proto_data[img_path] = []
                proto_data[img_path].append(box)
        return proto_data


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

    def __init__(self, encoding: str = 'utf-8') -> None:
        """ x, y, width, height """
        self._anno_path: Optional[pathlib.Path] = None
        self._imgs_folder: Optional[pathlib.Path] = None
        super().__init__(encoding)

    @property
    def imgs_folder(self) -> pathlib.Path:
        """ return folder which has images"""
        assert self._imgs_folder is not None
        return self._imgs_folder

    @imgs_folder.setter
    def imgs_folder(self, path: str) -> None:
        self._imgs_folder = pathlib.Path(path)

    @property
    def anno_path(self) -> pathlib.Path:
        """ path to the annotation.json file """
        assert self._anno_path is not None
        return self._anno_path

    @anno_path.setter
    def anno_path(self, path: str) -> None:
        self._anno_path = pathlib.Path(path)

    def fix_img_path(self, img_path: str) -> str:
        """ Fix image location according to provided folder """
        assert self._imgs_folder is not None
        path = pathlib.PureWindowsPath(img_path)
        my_path = self._imgs_folder / path.name
        return my_path.as_posix()

    def parse_anno(self) -> None:
        assert self._anno_path is not None
        with open(self._anno_path.as_posix()) as data_file:
            data = json.load(data_file, encoding=self.encode_method)
        for item in data:
            img_path = item['imageFilename']
            img_path = self.fix_img_path(img_path)
            boxes: List[BOX_TYPE] = []
            for box in item['Nodule']:
                x_1, y_1, _w, _h = box
                # Python count from 0 and slide omit last elem
                boxes.append(
                    BOX_TYPE(x_1 - 1, y_1 - 1, x_1 + _w, y_1 + _h, '1'))
            self.shapes.append(AnnosPerImage(boxes, img_path))


class XmlParser(AnnoParser):
    """ xml parser for standard VOC 2007 format """

    def __init__(self, encoding: str = 'utf-8') -> None:
        self.verified: bool = False
        self._folder_path: Optional[pathlib.Path] = None
        self.file_paths: List[pathlib.Path] = []
        super().__init__(encoding)

    @property
    def folder_path(self) -> pathlib.Path:
        """ return folder path, note: we use indirection here """
        assert self._folder_path is not None
        return self._folder_path

    @folder_path.setter
    def folder_path(self, path: str) -> None:
        """ set path to folder with .xml file """
        self._folder_path = pathlib.Path(path)
        self.file_paths = sorted(list(self._folder_path.glob('*.xml')))

    @staticmethod
    def add_shape(bndbox, label: str) -> BOX_TYPE:
        """ each line adds (x1, y1, x2, y2, label) """
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return BOX_TYPE(xmin, ymin, xmax, ymax, label)

    def img_abs_path(self, img_rel_path):
        """ return absolute path of image, for traing """
        temp_path = pathlib.PureWindowsPath(img_rel_path)
        img_rel_path = temp_path.as_posix()
        img_path = self._folder_path / img_rel_path

        return str(img_path.resolve())

    def parse_anno(self) -> None:
        for filepath in self.file_paths:
            parser = etree.XMLParser(encoding=self.encode_method)
            xmltree = ElementTree.parse(
                filepath.as_posix(), parser=parser).getroot()
            path_elem = xmltree.find('path')
            assert path_elem is not None
            img_rel_path = path_elem.text
            img_path = self.img_abs_path(img_rel_path)
            try:
                verified = xmltree.attrib['verified']
                self.verified = bool(verified)
            except KeyError:
                self.verified = False
            bboxes: List[BOX_TYPE] = []
            for object_iter in xmltree.findall('object'):
                bndbox = object_iter.find('bndbox')
                name_elem = object_iter.find('name')
                assert name_elem is not None
                label = name_elem.text
                assert label is not None
                my_box = self.add_shape(bndbox, label)
                # Assume dificulty False
                bboxes.append(my_box)
            self.shapes.append(AnnosPerImage(bboxes, img_path))


class AnnoWriter(metaclass=ABCMeta):
    """ Interface for Annotation Writter classes """

    def __init__(self, parser_obj: AnnoParser, binary: bool = False) -> None:
        self.data = parser_obj.anno_data
        self.binary = binary

    @abstractmethod
    def write(self) -> None:
        """ Writing down the annotations (either to file or folder) """
        pass

    def binary_transform(self, box: BOX_TYPE) -> BOX_TYPE:
        """ Transform box with label > 1 into label = 1 """
        label = "1" if self.binary and int(box.label) > 1 else box.label
        return BOX_TYPE(box.x_1, box.y_1, box.x_2, box.y_2, label)


class CSVWriter(AnnoWriter):
    """ Implementation for writting Retinanet CSV """

    def __init__(self, parser_obj, binary=False) -> None:
        super().__init__(parser_obj, binary)
        self.filename: Optional[str] = None

    def set_output_file(self, file_name) -> None:
        """ Set file to store the annotations """
        self.filename = file_name

    def write(self) -> None:
        """ Implementation for writting annos """
        assert self.filename is not None
        with open(self.filename, 'w', newline='') as csvfile:
            # anno_writer = csv.writer(
            #     csvfile,
            #     delimiter=',',
            #     quotechar='"',
            #     quoting=csv.QUOTE_MINIMAL)
            anno_writer = csv.writer(csvfile, dialect='excel')
            data_to_write: List[Tuple[str, int, int, int, int, str]] = []
            for img_file in self.data:
                for box in img_file.bbox:
                    box = self.binary_transform(box)
                    if box[4] == '0':
                        continue
                    data_to_write.append((img_file.img_path, box.x_1, box.y_1,
                                          box.x_2, box.y_2, box.label))
            for line in data_to_write:
                anno_writer.writerow(line)


class XMLWriter(AnnoWriter):
    """ Implementation for writting VOC .xml to destinated folder
    Very heavy since images are open to find size """

    def __init__(self, parser_obj: AnnoParser, binary: bool) -> None:
        super().__init__(parser_obj, binary)
        self.out_folder: Optional[pathlib.Path] = None
        self.img_vs_boxes = self._transform_data_to_dict()

    def set_output_folder(self, out_folder: str) -> None:
        """ Set file to store the annotations """
        self.out_folder = pathlib.Path(out_folder)
        if not self.out_folder.exists():
            raise ValueError("Output folder doesn't exist")

    def _transform_data_to_dict(self) -> ImageAnnoDict:
        """ Transform data to dict of anno according to img name """
        img_vs_boxes: ImageAnnoDict = {}
        for img_file in self.data:
            for box in img_file.bbox:
                box = self.binary_transform(box)
                if box[4] == '0':
                    continue
                if img_file.img_path not in img_vs_boxes:
                    img_vs_boxes[img_file.img_path] = []
                img_vs_boxes[img_file.img_path].append(box)
        return img_vs_boxes

    def write(self) -> None:
        for img_path, boxes in self.img_vs_boxes.items():
            path = pathlib.Path(img_path)
            if not path.exists():
                warnings.warn(
                    "File {} doesn't exists".format(path),
                    category=RuntimeWarning)
                continue
            img_folder_name = path.parts[-2]
            img_file_name = path.parts[-1]
            assert self.out_folder is not None
            new_file_path = create_corresponding_file(self.out_folder, path,
                                                      '.xml')
            xml_file = str(new_file_path)
            img = cv2.imread(str(path))    # very heavy
            print("Reading file {} to find its dimentions, please wait...".
                  format(img_file_name))
            img_shape = list(img.shape)
            write_pascal_voc(img_folder_name, img_file_name, img_shape,
                             xml_file, str(path), boxes)


class PascalVocWriter:
    """ Private writer for Pascal data format """

    def __init__(self, foldername: str, filename: str, imgSize: List[int],
                 xmlFile: str, localImgPath: str) -> None:
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = "Unknown"
        self.imgSize = imgSize
        self.boxlist: List[PASCAL_BOX_TYPE] = []
        self.localImgPath = localImgPath
        self.verified = False
        self.xmlFile = xmlFile

    def prettify(self, elem: Element) -> str:
        """ Return a pretty-printed XML string for the Element """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(
            root, pretty_print=True, encoding='utf-8').replace(
                "  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self) -> Element:
        """ Return XML root """
        # # Check conditions
        # if self.filename is None or \
        #         self.foldername is None or \
        #         self.imgSize is None:
        #     return None
        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername
        filename = SubElement(top, 'filename')
        filename.text = self.filename

        # if self.localImgPath is not None:
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

    def addBndBox(self, xmin: int, ymin: int, xmax: int, ymax: int, name: str,
                  difficult: bool) -> None:
        """ add bounding gox to internal record """
        bndbox = PASCAL_BOX_TYPE(xmin, ymin, xmax, ymax, name, difficult)
        self.boxlist.append(bndbox)

    def appendObjects(self, top) -> None:
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = each_object.label
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(each_object.y_2) == int(self.imgSize[0]) or (int(
                    each_object.y_1) == 1):
                truncated.text = "1"    # max == height or min
            elif (int(each_object.x_2) == int(self.imgSize[1])) or (int(
                    each_object.x_1) == 1):
                truncated.text = "1"    # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str(each_object.difficulty & 1)
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object.x_1)
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object.y_1)
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object.x_2)
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object.y_2)

    def save(self, targetFile: str) -> None:
        root = self.genXML()
        self.appendObjects(root)
        # out_file = None
        # if targetFile is None:
        #     out_file = codecs.open(
        #         self.filename + '.xml', 'w', encoding='utf-8')
        # else:
        out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult)
        out_file.close()


def create_corresponding_file(folder: pathlib.Path,
                              corresponding_file: pathlib.Path,
                              extension: str) -> pathlib.Path:
    """ Given: foler ('User/Desktop/'), file ('/yolo/foobar.jpg')
    extension('.xml'). Create: 'User/Desktop/foobar.xml"""
    orig_name_without_suffix = corresponding_file.stem
    new_name = orig_name_without_suffix + extension
    new_file = folder / new_name

    return new_file


def write_pascal_voc(img_folder_name: str, img_file_name: str,
                     img_shape: List[int], xml_file: str, img_path: str,
                     boxes: List[BOX_TYPE]) -> None:
    """ helper function to use PascalVocWriter class """
    writer = PascalVocWriter(
        img_folder_name,
        img_file_name,
        img_shape,
        xmlFile=xml_file,
        localImgPath=img_path)
    writer.verified = False
    for box in boxes:
        writer.addBndBox(box.x_1, box.y_1, box.x_2, box.y_2, box.label, False)
    writer.save(xml_file)
