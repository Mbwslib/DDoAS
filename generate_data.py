import json
import glob
import numpy as np
from PIL import Image


def main():
    cityscapes_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'rider',
                          'bus', 'train']
    prohibited_item_classes = ['Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors', 'Lighter', 'Battery', 'Bat', 'Razor_blade',
                               'Saw_blade', 'Fireworks', 'Hammer', 'Screwdriver', 'Dart', 'Pressure_vessel']
    files = glob.glob('./train_set/*.png')
    count = 0
    for file in files:
        #label_path = file[:-15] + 'gtFine_polygons.json'
        label_path = file[0:-3] + 'json'
        with open(label_path, 'r', encoding='utf8', errors='ignore') as j:
            label = json.load(j)
        Img = Image.open(file)
        w = Img.size[0]
        h = Img.size[1]
        items = label['objects']
        for single_prohibited_item in items:
            if single_prohibited_item['label'] in prohibited_item_classes:
                # image process
                min_b = np.min(np.array(single_prohibited_item['polygon']), axis=0)
                max_b = np.max(np.array(single_prohibited_item['polygon']), axis=0)
                single_prohibited_item_h = max_b[1] - min_b[1]
                single_prohibited_item_w = max_b[0] - min_b[0]
                h_extend = int(round(0.2 * single_prohibited_item_h))
                w_extend = int(round(0.2 * single_prohibited_item_w))
                min_w = np.maximum(0, min_b[0] - w_extend)
                min_h = np.maximum(0, min_b[1] - h_extend)
                max_w = np.minimum(w, max_b[0] + w_extend)
                max_h = np.minimum(h, max_b[1] + h_extend)
                single_prohibited_item_new_w = max_w - min_w
                single_prohibited_item_new_h = max_h - min_h

                scale_w = 224.0 / single_prohibited_item_new_w
                scale_h = 224.0 / single_prohibited_item_new_h
                new_single_prohibited_item = Img.crop(box=(min_w, min_h, max_w, max_h))
                new_single_prohibited_item = new_single_prohibited_item.resize((224, 224), Image.BILINEAR)
                new_single_prohibited_item.save('./trainset/' + str(count) + '.png')

                # label process
                dict = {}
                dict['label'] = single_prohibited_item['label']
                dict['polygon'] = []
                polygon_list = single_prohibited_item['polygon'][:]

                expend = True
                while(expend):
                    if len(polygon_list) < 60:
                        if len(polygon_list) < 31:
                            n = len(polygon_list)-1
                            for i in range(n):
                                new_w = (polygon_list[i][0] + polygon_list[i + 1][0]) / 2
                                new_h = (polygon_list[i][1] + polygon_list[i + 1][1]) / 2
                                single_prohibited_item['polygon'].insert((2 * i) + 1, [new_w, new_h])
                            polygon_list = single_prohibited_item['polygon'][:]
                        else:
                            n = 60 - len(polygon_list)
                            for i in range(n):
                                new_w = (polygon_list[i][0] + polygon_list[i+1][0]) / 2
                                new_h = (polygon_list[i][1] + polygon_list[i+1][1]) / 2
                                single_prohibited_item['polygon'].insert((2*i)+1, [new_w, new_h])
                            polygon_list = single_prohibited_item['polygon'][:]
                    else:
                        if len(polygon_list) == 60:
                            for point in single_prohibited_item['polygon']:
                                index_w = (point[0] - min_w) * scale_w
                                index_h = (point[1] - min_h) * scale_h
                                index_w = np.maximum(0, np.minimum(223, index_w))
                                index_h = np.maximum(0, np.minimum(223, index_h))
                                dict['polygon'].append([index_w, index_h])
                            expend = False
                        else:
                            scale = len(polygon_list) * 1.0 / 60
                            index_list = (np.arange(0, 60) * scale).astype(int)
                            for point in np.array(single_prohibited_item['polygon'])[index_list]:
                                index_w = (point[0] - min_w) * scale_w
                                index_h = (point[1] - min_h) * scale_h
                                index_w = np.maximum(0, np.minimum(223, index_w))
                                index_h = np.maximum(0, np.minimum(223, index_h))
                                dict['polygon'].append([index_w, index_h])
                            expend = False

                with open('./trainset/' + str(count) + '.json', 'w') as j:
                    json.dump(dict, j)
                count += 1
    print(count)


if __name__ == '__main__':
    main()





