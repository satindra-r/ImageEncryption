import numpy as np
import scipy.fftpack as fft
import PIL.Image as Image
from Crypto.Cipher import AES

fixed_point = 2 ** 15


def dct2(a):
	return fft.dct(fft.dct(a.T, norm='ortho').T, norm='ortho')


def idct2(a):
	return fft.idct(fft.idct(a.T, norm='ortho').T, norm='ortho')


def encrypt(key, img, scale):
	# subtract 127 so values are centered around 0
	dct_img = dct2(img - 127)

	# pad image width to multiple of 4 for AES
	size = dct_img.shape
	dct_img = np.pad(dct_img, ((0, 0), (0, (-size[1] % 4))))
	size = dct_img.shape

	# convert to fixed point for AES encryption
	dct_img_fixed = np.clip(dct_img * fixed_point, -2 ** 63, 2 ** 63 - 1).astype(np.int64)

	# encrypt only upper 32 bits as lower 32 bits may change due to floating point errors
	dct_img_fixed_msb = (dct_img_fixed >> 32).astype(np.int32)
	dct_img_fixed_lsb = dct_img_fixed.astype(np.uint32)
	dct_img_fixed_encrypted_msb = np.zeros(size).astype(np.int32)
	for i in range(size[0]):
		dct_bytes = dct_img_fixed_msb[i].tobytes()
		cipher = AES.new(key, AES.MODE_ECB)
		cipher_bytes = cipher.encrypt(dct_bytes)
		dct_img_fixed_encrypted_msb[i] = np.frombuffer(cipher_bytes, dtype=np.int32)
	dct_img_fixed_encrypted = dct_img_fixed_encrypted_msb.astype(np.int64) << 32 | dct_img_fixed_lsb.astype(np.int64)

	dct_img_encrypted = dct_img_fixed_encrypted.astype(np.float128) / fixed_point

	# simulate a scale down, in reality this would be done to the encrypted image
	dct_img_encrypted = dct_img_encrypted[:size[0] // scale, :]
	dct_img_encrypted = dct_img_encrypted[:, :size[1] // scale]

	# add bias back to image
	img_encrypted = idct2(dct_img_encrypted) + 127
	return img_encrypted


def decrypt(key, img_encrypted):
	# subtract 127 so values are centered around 0
	dct_img_encrypted = dct2(img_encrypted - 127)

	# remove columns to multiple of 4 for AES as padding will cause changes in last few frequencies while decrypting
	size = dct_img_encrypted.shape
	if (size[1] % 4):
		dct_img_encrypted = dct_img_encrypted[:, :-(size[1] % 4)]
	size = dct_img_encrypted.shape

	# convert to fixed point for AES decryption
	dct_img_fixed_encrypted = np.clip(dct_img_encrypted * fixed_point, -2 ** 63, 2 ** 63 - 1).astype(np.int64)

	# decrypt only upper 32 bits
	dct_img_fixed_encrypted_msb = (dct_img_fixed_encrypted >> 32).astype(np.int32)
	dct_img_fixed_lsb = dct_img_fixed_encrypted.astype(np.uint32)
	dct_img_fixed_msb = np.zeros(size).astype(np.int64)
	for i in range(dct_img_fixed_encrypted_msb.shape[0]):
		cipher_bytes = dct_img_fixed_encrypted_msb[i].tobytes()
		cipher = AES.new(key, AES.MODE_ECB)
		dct_bytes = cipher.decrypt(cipher_bytes)
		dct_img_fixed_msb[i] = np.frombuffer(dct_bytes, dtype=np.int32)
	dct_img_fixed = dct_img_fixed_msb.astype(np.int64) << 32 | dct_img_fixed_lsb.astype(np.int64)

	dct_img = dct_img_fixed.astype(np.float128) / fixed_point

	#add bias back to image
	img = idct2(dct_img) + 127
	return img


def main():
	input_path = input("Enter file path:")
	scale = int(input("Enter scale:"))
	key = (input("Enter key:") + "\0" * 16)[:16].encode("ascii")

	image = Image.open(input_path).convert('L')
	img = np.array(image).astype(np.float128)

	img_result = decrypt(key, encrypt(key, img, scale))

	image_result = Image.fromarray(np.clip(img_result, 0, 255).astype(np.uint8))
	output_path = input_path.replace(".png", "Result.png")
	image_result.save(output_path)

	image_smol = image.resize((image.width // scale, image.height // scale))
	output_path = input_path.replace(".png", "Scaled.png")
	image_smol.save(output_path)


if (__name__ == "__main__"):
	main()
