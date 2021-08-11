# importing google_images_download module
from bing_image_downloader.downloader import download

def downloadimages():

    search_queries = [
                      'Chlorophytum comosum', 'Epipremnum aureum',
                      'Cordyline australis', 'Spathiphyllum',
                      'Sansevieria zeylanica', 'Crassuwa ovata',
                      'Anthurium', 'Ficus lyrata',
                      'Monstera adansonii', 'Monstera deliciosa',
                      'Howea forsteriana', 'Aloe barbadensis miller'
                      ]

    for query in search_queries:
        download(
                 query, limit=1000,  output_dir='D:\\Data Warehouse\\plantabit\\0_rawdata',
                 adult_filter_off=True, force_replace=False,
                 timeout=60, verbose=True
                 )

    return


if __name__ == '__main__':
    downloadimages()
