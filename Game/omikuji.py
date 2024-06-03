import random

kuji = ["大吉","吉","中吉","小吉","末吉","凶","大凶"]

result = kuji[random.randrange(len(kuji))]
print(f"あなたの運勢は{result}です")