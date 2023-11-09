import rasterio
import os
import geojson
import shutil


TILE_SIZE_UTM = (1000,1000)
IMAGE_SIZE_PIXEL = (2500,2500)

class LoadMunich:

    def __init__(self,tif_folder,json_path,city):
        self.tif_folder_ = tif_folder
        self.json_path_ = json_path
        self.city_ = city

    def readTifKoos(self):
        id = 0
        out = {}
        for filename in os.listdir(self.tif_folder_):
            split_filename = filename.split('.')
            if split_filename[-1] == 'tif':
                tif_dataset = rasterio.open(f'{self.tif_folder_}/{filename}')
                bounds = tif_dataset.bounds #xmin=0,ymin=1,xmax=2,ymax=3 (utm32)
                one_tif = {'filename':filename,'xmin':bounds[0],'ymin':bounds[1],'xmax':bounds[2],'ymax':bounds[3]}
                out[id] = one_tif
                id += 1
        return out

    def readGeoJsonPoly(self):
        with open(self.json_path_) as f:
            gj = geojson.load(f)
        filepath_split = self.json_path_.split('/')
        filename = filepath_split[-1]
        out = {}
        for i in range(len(gj['features'])):
            poly_coos = gj['features'][i]['geometry']['coordinates'][0][0]
            out[i] = {'filename':filename,'polygon':poly_coos}
        return out

    def buildReadData(self, tif_koos,geojson_poly):
        polygon_images = {self.city_:{}}
        polygon_pixels = {}

        for t_id in tif_koos:
            for j_id in geojson_poly:
                intile = False
                #check if first polygon-point is in tile, y than x value
                #looping through coordinates of the polygon
                for i in range(len(geojson_poly[j_id]['polygon'])):
                    dy = geojson_poly[j_id]['polygon'][i][1]-tif_koos[t_id]['ymin']
                    if dy < TILE_SIZE_UTM[1] and dy >= 0:
                        dx = geojson_poly[j_id]['polygon'][i][0]-tif_koos[t_id]['xmin']
                        if dx < TILE_SIZE_UTM[0] and dx >= 0:
                            intile = True
                if intile:
                    #proofing if list exists --> append otherwise create
                    try:
                        polygon_images[self.city_][tif_koos[t_id]['filename']].append(j_id)
                    except:
                        polygon_images[self.city_][tif_koos[t_id]['filename']] = []
                        polygon_images[self.city_][tif_koos[t_id]['filename']].append(j_id)
                    polygon_pixels[j_id] = []
                    #looping through all coordinates per polygon
                    x_utm_2_pixel = IMAGE_SIZE_PIXEL[0]/TILE_SIZE_UTM[0]
                    y_utm_2_pixel = IMAGE_SIZE_PIXEL[1]/TILE_SIZE_UTM[1]
                    for i in range(len(geojson_poly[j_id]['polygon'])):
                            #tif origin lower left corner, np array origin upper left corner
                            #Transposed x <> y switched
                            polygon_pixels[j_id].append((abs((geojson_poly[j_id]['polygon'][i][1]-tif_koos[t_id]['ymin'])*y_utm_2_pixel-IMAGE_SIZE_PIXEL[1]),(geojson_poly[j_id]['polygon'][i][0]-tif_koos[t_id]['xmin'])*x_utm_2_pixel))
                                

        return polygon_images, polygon_pixels
    
    def copyTif(self,polygon_images):
        for image in polygon_images[self.city_]:
            try:
                shutil.copyfile(f"{self.tif_folder_}/{image}",f"data/masked_images_{self.city_}/{image}")
            except IOError:
                os.makedirs(os.path.dirname(f"data/masked_images_{self.city_}/"))
                shutil.copyfile(f"{self.tif_folder_}/{image}",f"data/masked_images_{self.city_}/{image}")

    def run(self):    
        tif_data = self.readTifKoos()
        json_data = self.readGeoJsonPoly()
        im, pix = self.buildReadData(tif_data,json_data)
        self.copyTif(im)
        return im, pix