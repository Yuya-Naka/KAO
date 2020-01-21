import requests
import os
import urllib
import math

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def create_image_path(save_dir_path, search_term, url):
    save_image_path = os.path.join(save_dir_path, search_term)
    make_dir(save_image_path)
    global image_number
    image_number += 1

    file_extension = os.path.splitext(url)[-1]
    if file_extension.lower() in (".jpg", ".jpeg", ".png"):
        full_path = os.path.join(save_image_path,
                                 str(image_number) + "_" + search_term + file_extension)
        return full_path
    else:
        raise ValueError("Not Applicable file extension")


def searching_image_by_q(url, headers, params, timeout=10):
    response = requests.get(url,
                            headers=headers,
                            params=params,
                            allow_redirects=True,
                            timeout=timeout)

    if response.status_code != 200:
        error = Exception("HTTP status: " + response.status_code)
        raise error

    return response


def validate_response_from_image_url(image_url):
    response = requests.get(image_url)

    content_type = response.headers['content-type']
    if "image" not in content_type:

        error = Exception("Content-Type: " + content_type)
        raise error

    return response


def save_image(filename, image):
    with open(filename, "wb") as f:
        f.write(image)

if __name__ == "__main__":
    SEARCH_TERM = "クリスティアーノロナウド"
    SEARCH_URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
    SUBSCRIPTION_KEY = "ecadd6d5e59a492cbc36127b0897a9d1"

    SAVE_DIR_PATH = "./saveimage"
    make_dir(SAVE_DIR_PATH)

    image_number = 0
    number_images_required = 300
    number_images_per_transaction = 150
    offset_count = math.floor(number_images_required / number_images_per_transaction)

    url_list = []

    headers = {
        'Content-Type': 'multipart/form-data',
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
    }

    print("Searching images for: ", SEARCH_TERM )

    for offset in range(offset_count):
        params = urllib.parse.urlencode({
            'q': SEARCH_TERM,
            'count': number_images_per_transaction,
            'offset': offset * number_images_per_transaction,
            'safeSearch': "Off",
        })

        try:
            response = searching_image_by_q(SEARCH_URL, headers, params)
            response_json = response.json()
        except Exception as err:
            print("[Error No.{0}] {1}".format(err.errno,
                                              err.strerror))
        else:
            for values in response_json['value']:
                img_url = urllib.parse.unquote(values['contentUrl'])
                if img_url:
                    url_list.append(img_url)
        for image_url in url_list:
            try:
                res = validate_response_from_image_url(image_url)

                image_path = create_image_path(SAVE_DIR_PATH,
                                               SEARCH_TERM,
                                               image_url)
                save_image(image_path, res.content)
                print("Saved image... {}".format(image_url))
            except KeyboardInterrupt:
                break
            except Exception as err:
                print("%s" % (err))
