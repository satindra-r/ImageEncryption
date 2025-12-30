import numpy as np
import scipy.fftpack as fft
import PIL.Image as Image
from Crypto.Cipher import AES


def dct2(a):
	return fft.dct(fft.dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
	return fft.idct(fft.idct(a.T, norm='ortho').T, norm='ortho')


def encrypt(key, img, scale):
	# subtract 127 so values are centered around 0
	dct_img = dct2(img - 127)

	# pad image width to multiple of 4 for AES
	size = dct_img.shape
	dct_img = np.pad(dct_img, ((0, 0), (0, (-size[1] % 8))))
	size = dct_img.shape

	dct_img_int_encrypted = np.full(size, 0x3FE << 52 | 0x1 << 34, dtype=np.uint64)
	cipher = AES.new(key, AES.MODE_ECB)
	for i in range(size[0]):
		for j in range(size[1] // 8):
			data = [np.uint64((dct_img[i][j * 8 + k].view(np.uint64) >> 48) & 0xFFFF) for k in range(8)]

			plain_bytes = np.array(data, dtype=np.uint16).tobytes()
			cipher_bytes = cipher.encrypt(plain_bytes)

			data_encrypted = np.frombuffer(cipher_bytes, dtype=np.uint16).astype(np.uint64)

			for k in range(8):
				dct_img_int_encrypted[i][j * 8 + k] |= data_encrypted[k] << 36

	dct_img_encrypted = np.frombuffer(dct_img_int_encrypted.tobytes(), dtype=np.float64).copy().reshape(*size)

	# add bias back to image
	img_encrypted = idct2(dct_img_encrypted) + 127
	return img_encrypted, dct_img_encrypted


def decrypt(key, img_encrypted, dct):
	# subtract 127 so values are centered around 0
	dct_img_encrypted = dct2(img_encrypted - 127)

	b = (dct_img_encrypted - dct)
	r = (dct_img_encrypted - dct) / dct
	# remove columns to multiple of 4 for AES as padding will cause changes in last few frequencies while decrypting
	size = dct_img_encrypted.shape
	if (size[1] % 8):
		dct_img_encrypted = dct_img_encrypted[:, :-(size[1] % 8)]
	size = dct_img_encrypted.shape

	dct_img_int = np.zeros(size, dtype=np.uint64)
	cipher = AES.new(key, AES.MODE_ECB)
	for i in range(size[0]):
		for j in range(size[1] // 8):
			data_encrypted = [np.uint64((dct_img_encrypted[i][j * 8 + k].view(np.uint64) >> 36) & 0xFFFF) for k in
							  range(8)]

			plain_bytes = np.array(data_encrypted, dtype=np.uint16).tobytes()
			cipher_bytes = cipher.decrypt(plain_bytes)
			data = np.frombuffer(cipher_bytes, dtype=np.uint16).astype(np.uint64)

			for k in range(8):
				dct_img_int[i][j * 8 + k] = data[k] << 48
	a = dct_img_int.tobytes()
	dct_img = np.frombuffer(dct_img_int.tobytes(), dtype=np.float64).copy().reshape(*size)

	# add bias back to image
	img = idct2(dct_img) + 127
	return img


def main():
	input_path = "images/img.png"  # input("Enter file path:")
	scale = 1  # int(input("Enter scale:"))
	key = ("1" + "\0" * 16)[:16].encode("ascii")  # (input("Enter key:") + "\0" * 16)[:16].encode("ascii")

	image = Image.open(input_path).convert('L')
	img = np.array(image).astype(np.float64)

	img_encrypted, a = encrypt(key, img, scale)

	image_encrypted = Image.fromarray(np.round(np.clip(img_encrypted, 0, 255)).astype(np.uint8))
	output_path = input_path.replace(".png", "Encrypted.png")
	image_encrypted.save(output_path)

	image_encrypted_scaled = image_encrypted.resize((image_encrypted.width // scale, image_encrypted.height // scale))
	output_path = input_path.replace(".png", "EncryptedScaled.png")
	image_encrypted_scaled.save(output_path)

	#img_encrypted_scaled = np.array(image_encrypted_scaled).astype(np.float64)
	#img_encrypted_scaled = img_encrypted.astype(np.float32).astype(np.float64)
	img_encrypted_scaled = img_encrypted
	#print(img_encrypted_scaled[0][0])
	#img_encrypted_scaled[0][0] = 231.377#np.round(img_encrypted_scaled[0][0])
	#print(img_encrypted_scaled[0][0])


	img_recovered = decrypt(key, img_encrypted_scaled, a)

	image_encrypted = Image.fromarray(np.clip(img_recovered, 0, 255).astype(np.uint8))
	output_path = input_path.replace(".png", "Recovered.png")
	image_encrypted.save(output_path)

	'''image_scaled = image.resize((image.width // scale, image.height // scale))
	output_path = input_path.replace(".png", "Scaled.png")
	image_scaled.save(output_path)'''


main()
