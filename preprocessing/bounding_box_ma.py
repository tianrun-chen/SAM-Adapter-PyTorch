import random

class BoundingBox:

    # borderspace could lead to boundingbox out of image borders --> exception?
    def buildFromPolygon(self, polygon_pixels, border_space = random.randrange(0,10)):
        bounding_box = {}
        for id, points in polygon_pixels.items():
            yvalues = []
            xvalues = []
            for coos in points:
                yvalues.append(coos[0])
                xvalues.append(coos[1])
            ymin = min(yvalues)
            xmin = min(xvalues)
            ymax = max(yvalues)
            xmax = max(xvalues)

            bounding_box[id] = (xmin-border_space,ymin-border_space,xmax+border_space,ymax+border_space)
        return bounding_box

    def getEdges(self, bounding_box):
        edges = {}
        for id, box in bounding_box.items():
            #[p0,p1,p2,p3] clockwise
            edges[id] = [(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]
        return edges
            