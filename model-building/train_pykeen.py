from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline

tf = TriplesFactory.from_path(
    "generated-files/triples.tsv",
    create_inverse_triples=True,
)

print(tf)

training, testing = tf.split([0.8, 0.2], random_state=42)

result = pipeline(
    training=training,
    testing=testing,
    model="TransE",
    epochs=20,
    model_kwargs=dict(
        embedding_dim=64,
    ),
    training_kwargs=dict(
        batch_size=1024,
    ),
)

result.save_to_directory("pykeen-lanl-model")