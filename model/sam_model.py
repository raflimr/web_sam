import numpy as np
import rasterio
from segment_anything import sam_model_registry, SamPredictor
from skimage.transform import resize
from rasterio.transform import from_origin


class SAMModel:
    def __init__(self, checkpoint_path, model_type="vit_b"):
        """
        Inisialisasi model SAM.
        
        Args:
            checkpoint_path (str): Path ke model checkpoint.
            model_type (str): Tipe model yang digunakan (default: "vit_b").
        """
        # Muat model SAM
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device='cpu')  # Gunakan CPU (atau ganti dengan GPU jika diperlukan)
        self.predictor = SamPredictor(self.sam)

    def read_multiband_tiff(self, image_path, bands_to_use=None, max_image_size=2048):
        """
        Membaca file TIFF multi-band dan mengembalikannya sebagai gambar 3 saluran (RGB).

        Args:
            image_path (str): Path ke file TIFF.
            bands_to_use (list): Indeks band yang akan digunakan (default: 3 saluran).
            max_image_size (int): Ukuran gambar maksimum yang diinginkan untuk resizing (default: 2048).

        Returns:
            np.ndarray: Gambar yang dibaca dan diresize (jika perlu).
        """
        with rasterio.open(image_path) as src:
            num_bands = src.count
            width = src.width
            height = src.height

            if bands_to_use is None:
                band_indices = [1, num_bands // 4, num_bands // 2, num_bands - 1]
                bands_to_use = band_indices[:3]

            image_bands = []
            for band_idx in bands_to_use:
                band = src.read(band_idx)
                band = ((band - band.min()) / (band.max() - band.min()) * 255).astype(np.uint8)
                image_bands.append(band)

            image = np.stack(image_bands, axis=-1)
            scale_factor = min(max_image_size / width, max_image_size / height)

            if scale_factor < 1:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = resize(image, (new_height, new_width), anti_aliasing=True, preserve_range=True).astype(np.uint8)

            return image

    def segment_image(self, image_path):
        """
        Melakukan segmentasi pada gambar menggunakan model SAM.

        Args:
            image_path (str): Path ke gambar yang akan diproses.

        Returns:
            np.ndarray: Mask hasil segmentasi.
        """
        # Membaca dan mempersiapkan gambar untuk segmentasi
        image = self.read_multiband_tiff(image_path)
        self.predictor.set_image(image)

        # Tentukan titik koordinat untuk segmentasi (tengah gambar)
        h, w = image.shape[:2]
        points = np.array([[w // 2, h // 2]])  # Titik tengah gambar
        labels = np.array([1])  # Foreground

        # Prediksi mask dengan model SAM
        masks, _, _ = self.predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)

        return masks

    def get_lat_lon(self, image_path, row, col):
        """
        Mengambil nilai latitude dan longitude berdasarkan posisi piksel dalam gambar.
        
        Args:
            image_path (str): Path ke file TIFF.
            row (int): Indeks baris piksel.
            col (int): Indeks kolom piksel.

        Returns:
            tuple: Latitude dan longitude pada posisi piksel tersebut.
        """
        with rasterio.open(image_path) as src:
            # Mendapatkan transformasi dari piksel ke koordinat dunia
            transform = src.transform
            
            # Menghitung koordinat dunia berdasarkan baris dan kolom
            lon, lat = transform * (col, row)
        
        return lat, lon