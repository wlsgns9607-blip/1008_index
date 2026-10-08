"""노트북 6개를 읽어 data.json을 만들고 웹사이트를 실행합니다.
사용법:  python app.py   →  http://localhost:8000
노트북(.ipynb)은 이 폴더 또는 ./notebooks 폴더에 두세요."""
import glob, json, os, re, http.server, socketserver

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 1) 노트북 코드 셀 정리 ----------
def clean(src):
    out = []
    for line in src.split("\n"):
        if line.strip().startswith("#") or "http" in line:
            out.append(line.rstrip()); continue
        l = re.sub(r"\s+([.,()\[\]:])", r"\1", line)   # 기호 앞 공백 제거
        l = re.sub(r"([(\[.])\s+", r"\1", l)           # 기호 뒤 공백 제거
        l = re.sub(r"\s*=\s*", "=", l)
        l = re.sub(r",(?=\S)", ", ", l)
        l = re.sub(r"^(\s*\w+)=(?!=)", r"\1 = ", l)    # 맨 앞 대입문만 보기 좋게
        out.append(l.rstrip())
    return "\n".join(out)

def read_code(n):
    files = glob.glob(os.path.join(HERE, f"**/*practice{n}-*.ipynb"), recursive=True)
    if not files: return []
    cells = []
    for c in json.load(open(files[0], encoding="utf-8"))["cells"]:
        s = "".join(c["source"]).strip()
        if c["cell_type"] != "code" or not s or s.startswith("!pip"): continue
        cap = s.split("\n")[0].lstrip("# ").strip() if s.startswith("#") else ""
        cells.append({"caption": cap, "src": clean(s)})
    return cells

# ---------- 2) 모델 shape / 파라미터 계산 ----------
def build(name, inp, specs):
    shape, layers, total = list(inp), [], 0
    for sp in specs:
        k = sp[0]; p = 0
        if k == "conv":
            _, f, ks, pad = sp
            h, w, c = shape
            p = f * (ks * ks * c + 1)
            if pad == "valid": h, w = h - ks + 1, w - ks + 1
            shape = [h, w, f]; desc = f"Conv2D {f}개 · {ks}×{ks}" + (" · same" if pad == "same" else "")
        elif k == "pool":
            shape = [shape[0] // 2, shape[1] // 2, shape[2]]; desc = "MaxPool 2×2"
        elif k == "flatten":
            n = 1
            for d in shape: n *= d
            shape = [n]; desc = "Flatten"
        else:
            n = sp[1]; p = shape[0] * n + n; shape = [n]; desc = f"Dense {n}"
        total += p
        layers.append({"type": k, "desc": desc, "out": list(shape), "params": p})
    return {"name": name, "input": list(inp), "layers": layers, "total": total}

C = lambda f, k=5, pad="valid": ("conv", f, k, pad)
P, F, D = ("pool",), ("flatten",), lambda n: ("dense", n)
head = [F, D(84), D(10)]

PAGES = [
 {"id": 1, "label": "이미지 미리보기", "title": "샘플 이미지셋과 차원",
  "summary": "MNIST(손글씨)와 CIFAR10(사물) 이미지의 모양을 확인하고, 배열의 차원(shape)을 읽는 법을 익힙니다.",
  "points": ["MNIST: 흑백 28×28 이미지 60,000장", "CIFAR10: 컬러 32×32×3 이미지 50,000장 (10종류)",
             "정답(y)은 MNIST는 (60000,), CIFAR10은 (50000, 1) 로 모양이 다릅니다."],
  "dims": [{"n": "d1", "shape": [5]}, {"n": "d2", "shape": [4, 5]}, {"n": "d3", "shape": [3, 4, 5]}, {"n": "d4", "shape": [2, 3, 4, 5]}],
  "labels": [{"n": "mnist_y", "shape": "(60000,)"}, {"n": "cifar_y", "shape": "(50000, 1)"}]},
 {"id": 2, "label": "Flatten", "title": "Flatten Layer로 이미지 학습",
  "summary": "28×28 이미지를 784개 숫자로 펴는 일을 reshape로 직접 하거나, Flatten 층에 맡길 수 있습니다. 결과는 같습니다.",
  "points": ["reshape: 데이터를 미리 (60000, 784)로 변환", "Flatten: 모델 안에서 (28, 28)을 784로 변환 → 데이터 전처리 불필요"],
  "models": [build("reshape (Input 784)", [784], [D(84), D(10)]), build("Flatten (Input 28×28)", [28, 28], [F, D(84), D(10)])]},
 {"id": 3, "label": "Convolution", "title": "Convolution Layer",
  "summary": "5×5 필터가 이미지 위를 미끄러지며 특징 지도(feature map)를 만듭니다. 마우스를 왼쪽 이미지 위에 올려 보세요.",
  "points": ["필터 5×5, padding 없음 → 28 − 5 + 1 = 24", "Conv 3개 → Conv 6개 → Flatten → Dense 84 → Dense 10"],
  "models": [build("Conv ×2", [28, 28, 1], [C(3), C(6)] + head)]},
 {"id": 4, "label": "Pooling", "title": "Pooling Layer",
  "summary": "MaxPool은 2×2 칸마다 최댓값만 남겨 크기를 절반으로 줄입니다. 파라미터가 20만 개에서 약 1만 개로 줄어듭니다.",
  "points": ["Flatten 직전 크기가 20×20×6 → 4×4×6 으로 감소", "노트북의 직접 계산식: 84×(2400+1) + 10×(84+1) + 78 + 456"],
  "models": [build("Pooling 없음", [28, 28, 1], [C(3), C(6)] + head),
             build("Pooling 사용", [28, 28, 1], [C(3), P, C(6), P] + head)]},
 {"id": 5, "label": "LeNet-5", "title": "LeNet-5 · MNIST와 CIFAR10",
  "summary": "고전 CNN LeNet-5 구조를 그대로 만들어 MNIST(흑백)와 CIFAR10(컬러)에 적용합니다.",
  "points": ["Conv 6 → Pool → Conv 16 → Pool → Dense 120 → 84 → 10", "MNIST는 padding='same'으로 28×28을 유지, CIFAR10은 입력 채널이 3"],
  "models": [build("MNIST", [28, 28, 1], [C(6, 5, "same"), P, C(16), P, F, D(120), D(84), D(10)]),
             build("CIFAR10", [32, 32, 3], [C(6), P, C(16), P, F, D(120), D(84), D(10)])]},
  {"id": 6, "label": "내 이미지", "title": "내 이미지 사용 (notMNIST)",
  "summary": "직접 내려받은 PNG 폴더(A~J)를 읽어 학습 데이터로 만들고 LeNet-5로 분류합니다.",
  "points": ["이미지 18,724장 · 28×28 · 10개 클래스 (폴더 이름이 정답)"],
  "classes": list("ABCDEFGHIJ"),
  "steps": ["glob으로 PNG 경로 모으기", "np.random.permutation으로 섞기", "plt.imread로 이미지 읽기",
            "폴더 이름을 정답(dependent)으로", "reshape(18724, 28, 28, 1)", "pd.get_dummies로 원-핫 인코딩"],
  "models": [build("LeNet-5", [28, 28, 1], [C(6, 5, "same"), P, C(16), P, F, D(120), D(84), D(10)])]},
 {"id": 7, "label": "신규 5종 이미지 분석", "title": "사용자 추가 이미지 5종 딥러닝 AI 정밀 분석",
  "summary": "도쿄 타워, 오사카성, 데덴네, 헵파이브 관람차, 교토 금각사 등 5장의 고해상도 이미지를 7번 탭에서 독립적으로 정밀 분석하고 시각화합니다.",
  "points": ["도쿄 타워 (IMG_5753.jpeg): 332.9m 에펠탑 양식 격자 철골 및 오렌지 배색 검출",
             "오사카성 천수각 (IMG_8848.jpeg): 전통 다층 기와 및 거석 석벽(이시가키) 매칭",
             "데덴네 (IMG_9054.jpeg): 포켓몬스터 6세대 안테나포켓몬 전기 볼주머니 특징",
             "헵파이브 대관람차 (IMG_9684.jpeg): 우메다 빌딩 옥상 크림슨 레드 휠 야경 실루엣",
             "킨카쿠지 금각사 (IMG_9703.jpeg): 경호지 수면 반사 및 순금박 누각 황금색 반사광"]},
]

def make_data():
    for p in PAGES:
        p["code"] = read_code(p["id"])
    json.dump({"pages": PAGES}, open(os.path.join(HERE, "data.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("data.json 생성 완료")

if __name__ == "__main__":
    make_data()
    os.chdir(HERE)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", 8000), http.server.SimpleHTTPRequestHandler) as s:
        print("http://localhost:8000  (종료: Ctrl+C)")
        s.serve_forever()
