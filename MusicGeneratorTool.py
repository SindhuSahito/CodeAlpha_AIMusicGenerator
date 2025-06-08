import os
import glob
import numpy as np
import random
from music21 import converter, instrument, note, chord, stream, tempo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def collect_midi_data(folder_path=r"D:\.vscode\New.py\Internship\midi_songs"):
    notes = []
    midi_files = glob.glob(os.path.join(folder_path, "*.mid")) + glob.glob(os.path.join(folder_path, "*.midi"))

    if not midi_files:
        raise FileNotFoundError("No MIDI files found in the specified folder.")

    print(f"Found {len(midi_files)} MIDI files")

    for file in midi_files:
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts if parts else midi.flat.notes

            for el in elements.stream().recurse():
                if isinstance(el, note.Note):
                    notes.append(str(el.pitch))
                elif isinstance(el, chord.Chord):
                    notes.append('.'.join(str(n) for n in el.normalOrder))
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

    print(f"Collected {len(notes)} notes/chords")
    return notes

def preprocess_data(notes, sequence_length=100, test_size=0.2):
    pitch_names = sorted(set(notes))
    note_to_int = {note: i for i, note in enumerate(pitch_names)}
    n_vocab = len(pitch_names)

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in sequence_in])
        network_output.append(note_to_int[sequence_out])

    X = np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(n_vocab)
    y = to_categorical(network_output, num_classes=n_vocab)

    return train_test_split(X, y, test_size=test_size), n_vocab, note_to_int

def build_model(input_shape, n_vocab):
    model = Sequential([
        LSTM(512, return_sequences=True, input_shape=input_shape),
        BatchNormalization(), Dropout(0.3),
        LSTM(512, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(512),
        BatchNormalization(), Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def train_model(model, X_train, X_val, y_train, y_val, epochs=5):
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = ModelCheckpoint(
        "checkpoints/model-{epoch:02d}.keras",
        monitor='val_loss',
        save_best_only=True
    )
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=64, callbacks=[checkpoint])

def generate_music(model, network_input, note_to_int, n_vocab, sequence_length=100, generation_length=500):
    int_to_note = {i: n for n, i in note_to_int.items()}
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    output = []

    for _ in range(generation_length):
        input_seq = np.reshape(pattern, (1, sequence_length, 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)
        index = np.random.choice(range(n_vocab), p=(prediction[0] / np.sum(prediction[0])))
        output.append(int_to_note[index])
        pattern = np.append(pattern, index)[1:]

    return output

def create_midi(predicted_notes, filename="generated_music.mid"):
    offset = 0
    output_notes = []

    for pattern in predicted_notes:
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.insert(0, tempo.MetronomeMark(number=120))
    midi_stream.write('midi', fp=filename)
    print(f"\nMIDI file saved as {filename}")

def main():
    try:
        notes = collect_midi_data()
        (X_train, X_val, y_train, y_val), n_vocab, note_to_int = preprocess_data(notes)
        model = build_model((X_train.shape[1], X_train.shape[2]), n_vocab)
        train_model(model, X_train, X_val, y_train, y_val, epochs=5)
        generated = generate_music(model, X_train, note_to_int, n_vocab)
        create_midi(generated)
        print("\nMusic generation complete!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
