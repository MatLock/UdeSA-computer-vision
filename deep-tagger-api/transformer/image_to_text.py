from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='es')
result = ocr.predict(input="/Users/matlock/Downloads/23353720619_011_00001_00000053.pdf")

for res in result:
    for text, score in zip(res['rec_texts'], res['rec_scores']):
        print(f"{text}\t{float(score):.4f}")