from typing import List, Any
import os
import glob as glob
from natsort import natsorted
import xml.etree.ElementTree as ET



def get_sort_string(conditions):
    string = []
    for i in range(len(conditions["compounds"])):
        compound = conditions["compounds"][i]
        concetration = conditions["concentrations"][i]
        unit = conditions["units"][i]
        string.append(f"{compound}{concetration}{unit}")
    string = sorted(string, key=lambda x: x)

    return "".join(string)


def sort_platemap(platemap):
    for i in range(96):
        sort_string = get_sort_string(platemap[i])
        platemap[i]["sorter"] = sort_string
    sorted_data_desc = sorted(platemap, key=lambda x: x["sorter"], reverse=True)
    return sorted_data_desc

def well_to_index(well):
    well = well.upper()
    row = ord(well[0]) - ord('A')
    col = int(well[1:]) - 1
    if 0 <= row < 8 and 0 <= col < 12:
        return row * 12 + col
    else:
        raise ValueError("Invalid well index")


def index_to_well(index):
    if 0 <= index < 96:
        row = index // 12
        col = index % 12
        return chr(row + ord('A')) + str(col + 1)
    else:
        raise ValueError("Index out of range (0 to 95)")


class PlateMap:
    uniqueTimestamps: list[Any]

    def __init__(self, path):
        self.root_path = path
        self.plate_map = []  # creates a platemap instance
        self.uniqueTimestamps = []  # Unique timestamps found in the data

        self.iter_index = 0
        self.platemap_name = os.path.basename(os.path.normpath(path))
        self.read_platemap()
        self.add_images_to_platemap()

    def get_platemap_name(self):
        return self.platemap_name
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.plate_map)

    def __next__(self):
        if self.iter_index < len(self.plate_map):
            well = self.plate_map[self.iter_index]
            self.iter_index += 1
            return well
        else:
            self.iter_index = 0
            raise StopIteration

    def get_unique_timestamps(self):
        return self.uniqueTimestamps

    def read_platemap(self):
        platefilename = ""
        for filename in os.listdir(self.root_path):
            if filename.endswith(".PlateMap"):
                platefilename = filename

        plate_path = os.path.join(self.root_path, platefilename)
        try:
            tree = ET.parse(plate_path)
        except FileNotFoundError as e:
            raise Exception(f"Platemap file not found from the folder! {e}")

        root = tree.getroot()
        platemap = []
        for well_num in range(0, 96):
            # Create a dictionary for the current well
            compounds: List[Any] = []
            well_dict = {
                'empty_well': True,
                'well_name': well_num,
                'cell_line': "ns",  # default is not spesified
                'cell_density': 0,  # default is 0
                'compounds': compounds,
                'concentrations': [],
                'units': [],
                'sorter': "",

                'image_paths': [],
                'results': [],  # Results are stored as tuplets (norm, ferr) in the list
                'fl_image_paths': [],
                'geojson_paths': [],
                'fl_results': [],

            }
            platemap.append(well_dict)

        wells = root.find("wellStore/wells")
        for well in wells:
            index = int(well.get("row")) * 12 + int(well.get("col"))
            if len(well) > 0:
                for item in well.find("items"):
                    if item.get("type") == "Compound":
                        concentration = item.get("concentration")
                        units = item.get("concentrationUnits")
                        drug = item.find("referenceItem").get("displayName")
                        platemap[index]["compounds"].append(drug)
                        platemap[index]["concentrations"].append(concentration)
                        platemap[index]["units"].append(units)

                    elif item.get("type") == "CellType":
                        platemap[index]["cell_line"] = item.find("referenceItem").get("displayName")
                        platemap[index]["cell_density"] = int(item.get("seedingDensity"))
        self.plate_map = platemap

    def add_images_to_platemap(self):
        image_paths = natsorted(glob.glob(self.root_path + "/*.png"))
        fl_paths = natsorted(glob.glob(self.root_path + "/*.tif"))
        geojson_paths = natsorted(glob.glob(self.root_path + "/*.geojson"))
        unique_timestamps = []
        highest_ind = 0
        for name in glob.glob(self.root_path + "/*"):
            if "PlateMap" in name or os.path.basename(name) == "":
                continue
            basename = os.path.basename(name)
            parsed_string = basename.split('_')
            days = int(parsed_string[3][0:2])
            hours = int(parsed_string[3][3:5])
            minutes = int(parsed_string[3][6:8]) + days * 1440 + hours * 60
            img_ind = int(parsed_string[2])
            if highest_ind < img_ind:
                highest_ind = img_ind
            if minutes not in unique_timestamps:
                unique_timestamps.append(minutes)
        unique_timestamps.sort()
        self.uniqueTimestamps = unique_timestamps

        for well in self.plate_map:
            well['image_paths']: List[Any] = [[None for j in range(highest_ind)] for i in range(len(unique_timestamps))]
            well['results']: List[Any] = [[None for j in range(highest_ind)] for i in range(len(unique_timestamps))]
            well['fl_results']: List[Any] = [[None for j in range(highest_ind)] for i in range(len(unique_timestamps))]
            well['fl_image_paths']: List[Any] = [[None for j in range(highest_ind)] for i in range(len(unique_timestamps))]
            well['geojson_paths']: List[Any] = [[None for j in range(highest_ind)] for i in range(len(unique_timestamps))]

        if len(image_paths) > 0:
            # Loop image paths and put them in the wells accordingly
            for i, name in enumerate(image_paths):
                basename = os.path.basename(name)
                parsed_string = basename.split('_')
                plate_name = parsed_string[0]
                well = well_to_index(parsed_string[1])
                img_ind = int(parsed_string[2]) - 1
                days = int(parsed_string[3][0:2])
                hours = int(parsed_string[3][3:5])
                minutes = int(parsed_string[3][6:8]) + days * 1440 + hours * 60

                self.plate_map[well]['image_paths'][unique_timestamps.index(minutes)][img_ind] = name
                self.plate_map[well]['empty_well'] = False

        if len(fl_paths) > 0:
            # Loop fl paths and put them in the wells accordingly
            for i, name in enumerate(fl_paths):
                basename = os.path.basename(name)
                parsed_string = basename.split('_')
                plate_name = parsed_string[0]
                well = well_to_index(parsed_string[1])
                img_ind = int(parsed_string[2]) - 1
                days = int(parsed_string[3][0:2])
                hours = int(parsed_string[3][3:5])
                minutes = int(parsed_string[3][6:8]) + days * 1440 + hours * 60

                self.plate_map[well]['fl_image_paths'][unique_timestamps.index(minutes)][img_ind] = name
                self.plate_map[well]['empty_well'] = False

        if len(geojson_paths) > 0:
            # Loop image paths and put them in the wells accordingly
            for i, name in enumerate(geojson_paths):
                basename = os.path.basename(name)
                parsed_string = basename.split('_')
                plate_name = parsed_string[0]
                well = well_to_index(parsed_string[1])
                img_ind = int(parsed_string[2]) - 1
                days = int(parsed_string[3][0:2])
                hours = int(parsed_string[3][3:5])
                minutes = int(parsed_string[3][6:8]) + days * 1440 + hours * 60

                self.plate_map[well]['geojson_paths'][unique_timestamps.index(minutes)][img_ind] = name
                self.plate_map[well]['empty_well'] = False

    def get_phase_image_path(self, well, timestamp, ind):
        return self.plate_map[well]['image_paths'][timestamp][ind]

    def insert_results(self, timestamp, index, results):
        self.plate_map[self.iter_index]['results'][timestamp][index] = results

    def get_results_iter(self):
        norm_stamps = [0 for j in range(len(self.uniqueTimestamps))]
        ferr_stamps = [0 for j in range(len(self.uniqueTimestamps))]
        empty = True
        for time, timestamp_list in enumerate(self.plate_map[self.iter_index-1]['results']):
            for ind, results in enumerate(timestamp_list):
                if results is not None:
                    norm_stamps[time] += results[0]
                    ferr_stamps[time] += results[1]
                    empty = False

        return norm_stamps, ferr_stamps, empty


    def get_condition_string_iter(self):
        conditions = self.plate_map[self.iter_index-1]
        string = []
        cell_line = conditions["cell_line"]

        if cell_line != "ns":
            string.append(cell_line)
            string.append(" ")

        for i in range(len(conditions["compounds"])):
            compound = conditions["compounds"][i]
            concetration = conditions["concentrations"][i]
            unit = conditions["units"][i]
            if i + 1 == len(conditions["compounds"]):
                string.append(f"{compound} {concetration} {unit} ")
            else:
                string.append(f"{compound} {concetration} {unit}, ")

        well = index_to_well(conditions["well_name"])
        string.append(f"({well})")
        return "".join(string)


if __name__ == '__main__':
    print()
