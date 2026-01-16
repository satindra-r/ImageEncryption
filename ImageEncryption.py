import numpy as np
import scipy.fftpack as fft
import PIL.Image as Image
from Crypto.Cipher import AES
from bitarray import bitarray

OFFSET = 6


def dct2(a):
	return fft.dct(fft.dct(a.T, norm='forward').T, norm='forward')


def idct2(a):
	return fft.idct(fft.idct(a.T, norm='backward').T, norm='backward')


def enc_float(f, OTP):
	OTP_bits = bitarray(endian="big")
	OTP_bits.frombytes(OTP)
	f_bits = bitarray(endian="big")
	f_bits.frombytes(f.view(dtype=np.dtype(">f4")).tobytes())
	f_bits_enc = bitarray(32, endian="big")
	f_bits_enc.setall(0)
	f_bits_enc[0] = f_bits[0] ^ OTP_bits[0]
	found_set = False
	for i in range(1, 1 + OFFSET):
		f_bits_enc[i] = f_bits[i]

	for i in range(2 + OFFSET, 32):
		if found_set:
			f_bits_enc[i] = f_bits[i - 1] ^ OTP_bits[i - 1]
		elif f_bits[i - 1]:
			f_bits_enc[i] = 1
			found_set = True
	return np.frombuffer(f_bits_enc.tobytes(), dtype=np.dtype(">f4"))[0]


def dec_float(f_enc, OTP):
	OTP_bits = bitarray(endian="big")
	OTP_bits.frombytes(OTP)
	f_bits_enc = bitarray(endian="big")
	f_bits_enc.frombytes(f_enc.view(dtype=np.dtype(">f4")).tobytes())
	f_bits = bitarray(32, endian="big")
	f_bits.setall(0)
	f_bits[0] = f_bits_enc[0] ^ OTP_bits[0]
	found_set = False

	for i in range(1, 1 + OFFSET):
		f_bits[i] = f_bits_enc[i]

	for i in range(2 + OFFSET, 32):
		if found_set:
			f_bits[i - 1] = f_bits_enc[i] ^ OTP_bits[i - 1]
		elif f_bits_enc[i]:
			f_bits[i - 1] = 1
			found_set = True

	return np.frombuffer(f_bits, dtype=np.dtype(">f4"))[0]


def encrypt_img(img, key):
	# subtract 127 so values are centered around 0
	dct_img = dct2(img - 127)
	img2 = idct2(dct_img) + 127
	dct_img2 = dct2(img2 - 127)

	size = dct_img.shape

	dct_img_encrypted = np.zeros(size, dtype=np.float32)
	cipher = AES.new(key, AES.MODE_CTR)
	maxSize = max(size[0], size[1])
	for i in range(maxSize):
		for j in range(i + 1):
			OTP = cipher.encrypt(b'\x00' * 4)
			if i < size[0] and j < size[1]:
				dct_img_encrypted[i][j] = enc_float(dct_img[i][j], OTP)
		for j in range(i - 1, -1, -1):
			OTP = cipher.encrypt(b'\x00' * 4)
			if i < size[1] and j < size[0]:
				dct_img_encrypted[j][i] = enc_float(dct_img[j][i], OTP)

	# add bias back to image
	img_encrypted = idct2(dct_img_encrypted) + 127
	return img_encrypted, cipher.nonce


def decrypt_img(img_encrypted, key, nonce):
	# subtract 127 so values are centered around 0
	dct_img_encrypted = dct2(img_encrypted - 127)
	size = dct_img_encrypted.shape

	dct_img = np.zeros(size, dtype=np.float32)
	cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
	maxSize = max(size[0], size[1])
	for i in range(maxSize):
		for j in range(i + 1):
			OTP = cipher.encrypt(b'\x00' * 4)
			if i < size[0] and j < size[1]:
				dct_img[i][j] = dec_float(dct_img_encrypted[i][j], OTP)
		for j in range(i - 1, -1, -1):
			OTP = cipher.encrypt(b'\x00' * 4)
			if i < size[1] and j < size[0]:
				dct_img[j][i] = dec_float(dct_img_encrypted[j][i], OTP)

	# add bias back to image
	img = idct2(dct_img) + 127
	return img


def encrypt(key, image):
	imager, imageg, imageb = image.convert("RGB").split()
	imgr = np.array(imager).astype(np.float32)
	imgg = np.array(imageg).astype(np.float32)
	imgb = np.array(imageb).astype(np.float32)

	imgr_encrypted, noncer = encrypt_img(imgr, key)
	imgg_encrypted, nonceg = encrypt_img(imgg, key)
	imgb_encrypted, nonceb = encrypt_img(imgb, key)

	imager_encrypted = Image.fromarray(np.round(np.clip(imgr_encrypted, 0, 255)).astype(np.uint8))
	imageg_encrypted = Image.fromarray(np.round(np.clip(imgg_encrypted, 0, 255)).astype(np.uint8))
	imageb_encrypted = Image.fromarray(np.round(np.clip(imgb_encrypted, 0, 255)).astype(np.uint8))

	image_encrypted = Image.merge("RGB", (imager_encrypted, imageg_encrypted, imageb_encrypted))
	return image_encrypted, (noncer, nonceg, nonceb)


def decrypt(key, image_encrypted, nonce):
	imager_encrypted, imageg_encrypted, imageb_encrypted = image_encrypted.convert("RGB").split()
	imgr_encrypted = np.array(imager_encrypted).astype(np.float32)
	imgg_encrypted = np.array(imageg_encrypted).astype(np.float32)
	imgb_encrypted = np.array(imageb_encrypted).astype(np.float32)

	imgr = decrypt_img(imgr_encrypted, key, nonce[0])
	imgg = decrypt_img(imgg_encrypted, key, nonce[1])
	imgb = decrypt_img(imgb_encrypted, key, nonce[2])

	imager = Image.fromarray(np.round(np.clip(imgr, 0, 255)).astype(np.uint8))
	imageg = Image.fromarray(np.round(np.clip(imgg, 0, 255)).astype(np.uint8))
	imageb = Image.fromarray(np.round(np.clip(imgb, 0, 255)).astype(np.uint8))

	image = Image.merge("RGB", (imager, imageg, imageb))
	return image


def main():
	input_path = "images/img.png"  # input("Enter file path:")
	scale = 2  # int(input("Enter scale:"))
	key = ("1" + "\0" * 16)[:16].encode("ascii")  # (input("Enter key:") + "\0" * 16)[:16].encode("ascii")

	image_encrypted, nonce = encrypt(key, Image.open(input_path))
	output_path = input_path.replace(".png", "Encrypted.png")
	image_encrypted.save(output_path)

	image_encrypted_scaled = image_encrypted.resize((image_encrypted.width // scale, image_encrypted.height // scale))
	output_path = input_path.replace(".png", "EncryptedScaled.png")
	image_encrypted_scaled.save(output_path)

	image_recovered = decrypt(key, image_encrypted_scaled, nonce)

	output_path = input_path.replace(".png", "Recovered.png")
	image_recovered.save(output_path)


main()
