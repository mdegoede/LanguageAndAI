from embeddings_functions import FasttextEmbedding
import fasttext

processor = FasttextEmbedding('data/birth_year.csv', 'result/trained_fasttext_embeddings.bin')
model = processor.train_fasttext_model()
# model = fasttext.load_model("result/trained_fasttext_embeddings.bin")
processor.process_and_add_embedding()
processor.calculate_and_add_average_column()
mil_and_genz = processor.mil_and_genz
mil_and_genz.to_csv('data/mil_and_genz.csv', index=False)
print(mil_and_genz, mil_and_genz.columns)