import os

def create_folders(type):

    folders = {'main':{'data':'data','pretrained':'pretrained'},
                    'custom':{'geojson':'data/munich_geojson','tiles':'data/geodatenbayern_munich_lk_city'},
                    'postprocessing':{'binary':'binary','hists':'hists','overlay':'overlay'}}
    
    if type == 'all':
        for type_group in folders:
            for folder in folders[type_group]:
                try:
                    os.mkdir(folders[type_group][folder])
                except:
                    print(f'Folder {folder} already exists!')

    else:
        for folder in folders[type]:
            try:
                os.mkdir(folders[type][folder])
            except:
                print(f'Folder {folder} already exists!')