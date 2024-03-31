from PIL import Image
import numpy as np

POSITIVE = 0
NEGATIVE = 1
RANDOM = 2
"""
    bin():turn number to binary numbers
        return string
    str.zfill(): fill the left of string with zeros, till the length specified
"""


def bits_fill(bits):
    bits_size = ((len(bits) + 7) // 8) * 8
    return bits.zfill(bits_size)


def text_to_bits(text: str, encode_type='utf-8'):
    bits = bin(int.from_bytes(text.encode(encode_type), 'big'))[2:]
    bits_size = ((len(bits) + 7) // 8) * 8
    return bits.zfill(bits_size)


def bits_to_text(bits: str, encode_type='utf-8'):
    n = int(bits, 2)
    bytes_number = (n.bit_length() + 7) // 8
    return n.to_bytes(bytes_number, 'big').decode(encode_type)


def generate_random_bits(length=100) -> str:
    """
        不会进行相应的字节填充
    """
    if length <= 0:
        return ''
    bits = np.random.randint(2, size=length)
    str_bits = ''
    for bit in bits:
        str_bits += str(bit)
    return str_bits


def lsb_steganography(image_path, bits: str, save_path, mask_type=0, seed=None):
    """
        set the least significant bit to zero and write the data

        while usually, lsb is:
            written data: n bits from {0, 1}, eg: 01011101010...
            for each pixel's least significant bit, do positive or negative masked, where mask = written data

        positive masked operator
            F_lsb = 0<->1 2<->3 ... 254<->255
        negative masked operator
            F_lsb_-1 = -1<->0 1<->2 ... 255<->256
    """
    default_rng_state = None
    if seed is not None:
        default_rng_state = np.random.get_state()
        np.random.seed(seed)
    image = Image.open(image_path)
    image_array = np.array(image.getdata())
    channels = image_array.shape[-1]
    image_array = image_array.flatten()
    # steganography
    bits = bits_fill(bits)
    bits_len = len(bits)
    num_pixels = image_array.shape[0]
    if num_pixels < bits_len:
        print('Data and flag are too long to embedded in image')
        return
    if mask_type == 2:  # random
        rand_index_array = np.arange(num_pixels)
        np.random.shuffle(rand_index_array)
        rand_index_array = rand_index_array[:bits_len]
        for i, index in enumerate(rand_index_array):
            image_array[index] = image_array[index] & ~1 | int(bits[i])
    else:
        for i in range(bits_len):
            if mask_type == 0:  # positive
                image_array[i] = image_array[i] & ~1 | int(bits[i])
            else:  # negative
                image_array[i] = (((image_array[i] + 1) & ~1) - int(bits[i])) % 256
    ret_image_array = image_array.reshape(*image.size[::-1], channels).astype(np.uint8)
    ret_image = Image.fromarray(ret_image_array)
    ret_image.save(save_path)
    if seed is not None:
        np.random.set_state(default_rng_state)


def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    if n < m:
        return -1
    prefix = get_prefix(pattern)
    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = prefix[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1


def get_prefix(pattern):
    m = len(pattern)
    prefix = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = prefix[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        prefix[i] = j
    return prefix


def extrac_data(image_path, flag, mask_type=0, seed=None):
    default_rng_state = None
    if seed is not None:
        default_rng_state = np.random.get_state()
        np.random.seed(seed)
    image = Image.open(image_path)
    embedded_image_array = np.array(image.getdata()).flatten()
    flag_bits = text_to_bits(flag)
    embedded_image_array = embedded_image_array & 1
    embedded_str = ''
    if mask_type == 2:  # random
        rand_index_array = np.arange(embedded_image_array.shape[0])
        np.random.shuffle(rand_index_array)
        for index in rand_index_array:
            embedded_str += str(embedded_image_array[index])
    else:
        for bit in embedded_image_array:
            embedded_str += str(bit)
    index = kmp_search(embedded_str, flag_bits)
    data_bits = embedded_str[:index]
    if seed is not None:
        np.random.set_state(default_rng_state)
    return data_bits


def lsb_text_not_encode(image_path, data: str, flag: str, save_path, mask_type=0, seed=None):
    # data and flag to bits
    data_bits = text_to_bits(data)
    flag_bits = text_to_bits(flag)
    bits = data_bits + flag_bits
    lsb_steganography(image_path, bits, save_path, mask_type, seed)


def extrac_lsb_text_not_encode(image_path, flag, mask_type=0, seed=None):
    data_bits = extrac_data(image_path, flag, mask_type, seed)
    return bits_to_text(data_bits)


if __name__ == '__main__':
    np.random.seed(42)
    lsb_text_not_encode(image_path='./images/miku.png', data='21312246, 林恒盛', flag='12345',
                        save_path='./secrets/miku.png', mask_type=RANDOM)
    np.random.seed(42)
    ret = extrac_lsb_text_not_encode('./secrets/miku.png', flag='12345', mask_type=RANDOM)
    print(ret)
