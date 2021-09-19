from PIL import Image
import os

folders = ['train', 'test', 'validation']
species = [
          'Chlorophytum comosum', 'Epipremnum aureum',
          'Cordyline australis', 'Spathiphyllum',
          'Sansevieria zeylanica', 'Crassuwa ovata',
          'Anthurium', 'Ficus lyrata',
          'Monstera adansonii', 'Monstera deliciosa',
          'Howea forsteriana', 'Aloe barbadensis miller'
          ]
for f in folders:
    for s in species:
        directory = 'D:\\Data Warehouse\\plantabit\\3_rawdata_clean_no_duplicates_aug\\{}\\{}'.format(f, s)
        c=1
        for filename in os.listdir(directory):
            print('\nReading file: {}'.format(os.path.join(directory, filename)))
            if filename.endswith(".png"):
                continue
            else:
                print('Converting {} into png'.format(filename))
                im = Image.open(os.path.join(directory, filename))
                name='img'+str(c)+'.png'
                rgb_im = im.convert('RGB')
                rgb_im.save(os.path.join(directory, name))
                c+=1
