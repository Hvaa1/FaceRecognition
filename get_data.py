import os
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

# Thư mục lưu ảnh
save_dir = "images"
os.makedirs(save_dir, exist_ok=True)

# Truy vấn SPARQL để lấy danh sách người Việt có ảnh trên Wikidata
endpoint_url = "https://query.wikidata.org/sparql"
query = """
SELECT ?person ?personLabel ?image WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "vi,en". }
  ?person wdt:P31 wd:Q5.
  ?person wdt:P27 wd:Q881.
  ?person wdt:P18 ?image.
}
"""

def get_results(endpoint_url, query):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.query().convert()
        return result.get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"Lỗi truy vấn SPARQL: {e}")
        return []

# Lấy danh sách người Việt có ảnh từ Wikidata
results = get_results(endpoint_url, query)

if not results:
    print("Không có kết quả nào được tìm thấy.")
else:
    for result in results:
        try:
            name = result["personLabel"]["value"]
            image_url = result["image"]["value"]

            # Tạo tên file hợp lệ
            filename = os.path.join(save_dir, name.replace(" ", "_") + ".jpg")

            # Tải ảnh về máy
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f" Đã tải: {name} -> {filename}")
            else:
                print(f" Lỗi tải ảnh cho {name}")

        except Exception as e:
            print(f"⚠ Bỏ qua lỗi với {name}: {e}")

print(" Hoàn tất tải ảnh!")
