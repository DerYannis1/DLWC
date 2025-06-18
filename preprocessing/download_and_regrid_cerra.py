import cdsapi
import argparse
import os

def download_cerra(name, month, year, save_dir):

    dataset = "reanalysis-cerra-pressure-levels"
    request = {
    "variable": [
        "geopotential",
        "relative_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "pressure_level": [
        "200", "300", "500",
        "700", "850", "925",
        "1000"
    ],
    "data_type": ["reanalysis"],
    "product_type": ["analysis"],
    "year": [year],
    "month": [month],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "03:00", "06:00",
        "09:00", "12:00", "15:00",
        "18:00", "21:00"
    ],
    "data_format": "grib"
    }

    client = cdsapi.Client()

    filename = os.path.join(save_dir, f"cerra_{name}_{year}.grib")
    client.retrieve(dataset, request).download(filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Download and regrid NetCDF files.')
    parser.add_argument('--years', type=str, nargs='+', required=True, help='Years to download')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save data.')
    return parser.parse_args()


def main():

    args = parse_args()

    years = args.years
    save_dir = args.save_dir

    for year in years:
        for i in range(12):
            month = str(i + 1).zfill(2)
            name = i + 1

            download_cerra(name, month, year, save_dir)
            print("Download completed. File saved as cerra_{}_{}.grib, for the month {}.".format(name, year, month))

            os.system("cdo remapbil,grid_cerra.txt {}/cerra_{}_{}.grib {}/cerra_remap_{}_{}.grib".format(save_dir, name, year, save_dir, name, year))
            os.system("cdo -f nc copy {}/cerra_remap_{}_{}.grib {}/cerra_{}_{}.nc".format(save_dir, name, year, save_dir, name, year))

            try:
                os.remove("{}/cerra_{}_{}.grib".format(save_dir, name, year))
                os.remove("{}/cerra_remap_{}_{}.grib".format(save_dir, name, year))
                print("Temporary .grib files removed for month {}.".format(month))
            except FileNotFoundError as e:
                print(f"Warning: could not delete file - {e}")


if __name__ == "__main__":
    main()
