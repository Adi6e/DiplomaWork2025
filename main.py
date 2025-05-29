import os
import numpy as np
import whisper
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pydub import AudioSegment
import pyaudio
import wave
from datetime import datetime
from pathlib import Path
from natasha import Segmenter, MorphVocab, NewsMorphTagger, NewsEmbedding, Doc
from utils import bad_words
import torch

# Инициализация компонентов Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Кэш для лемм
LEMMA_CACHE = {}

# Get the directory where the script is located - используем абсолютный путь
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to get absolute path for any file relative to the script directory
def get_abs_path(relative_path):
    return os.path.normpath(os.path.join(SCRIPT_DIR, relative_path))

# Update the ensure_directories function
def ensure_directories():
    """Создаем необходимые директории, если их нет"""
    dirs = ["Audio", "Audio/Input_audios", "Audio/Output_audios"]
    for dir_path in dirs:
        path = Path(get_abs_path(dir_path))
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"Создана директория: {path.absolute()}")
            else:
                print(f"Директория уже существует: {path.absolute()}")
        except Exception as e:
            print(f"Ошибка при создании директории {path}: {str(e)}")
            raise

def load_custom_whisper_model(model_path, log_callback=None):
    """Загружает кастомную файнтюнинговую модель whisper"""
    try:
        if log_callback:
            log_callback(f"Загрузка кастомной модели из {model_path}...")
        
        # Проверяем существование файла модели
        if not os.path.exists(model_path):
            if log_callback:
                log_callback(f"Файл модели не найден: {model_path}")
            return None
        
        # Загружаем базовую модель whisper-small
        base_model = whisper.load_model("small")
        
        # Загружаем веса файнтюнинговой модели
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        # Если checkpoint содержит состояние модели
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Загружаем состояние в модель
        base_model.load_state_dict(state_dict, strict=False)
        
        if log_callback:
            log_callback("Кастомная модель успешно загружена!")
        
        return base_model
        
    except Exception as e:
        if log_callback:
            log_callback(f"Ошибка при загрузке кастомной модели: {str(e)}")
        return None

# Функции обработки аудио из предыдущего кода
def generate_beep(duration_ms, frequency=1000, sample_rate=16000):
    """
    Генерация звукового сигнала (писка) с определенной частотой.
    """
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)  # Генерация синусоиды
    beep = (beep * 32767).astype(np.int16)  # Преобразуем в 16-битный формат
    beep_sound = AudioSegment(beep.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    return beep_sound

def jaccard_similarity(str1, str2):
    """
    Расчет коэффициента сходства Жаккарда между двумя строками.
    """
    set1, set2 = set(str1), set(str2)
    if not set1 or not set2:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def lemmatize_word(word):
    """
    Возвращает лемму слова с помощью Natasha, с кэшированием.
    """
    word = word.lower().strip()
    if word in LEMMA_CACHE:
        return LEMMA_CACHE[word]

    doc = Doc(word)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        if token.lemma:
            lemma = token.lemma.lower()
            LEMMA_CACHE[word] = lemma
            return lemma

    # Если лемма не найдена, возвращаем слово как есть
    LEMMA_CACHE[word] = word
    return word

def find_all_prohibited_lemmas(prohibited_words_list):
    """
    Лемматизирует запрещённые слова и возвращает их в виде множества.
    """
    return set(lemmatize_word(word) for word in prohibited_words_list)

def mute_prohibited_words(audio_path, output_path, prohibited_words_list, similarity_threshold=0.7, log_callback=None):
    if log_callback is None:
        log_callback = print
    
    # Проверяем пути на абсолютность и существование
    audio_path = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    
    # Создаем директорию для выходного файла, если нужно
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Пробуем загрузить кастомную модель
    custom_model_path = "./Model/whisper_golos.pt"
    model = load_custom_whisper_model(custom_model_path, log_callback)
    
    if model is None:
        log_callback("Не удалось загрузить кастомную модель, используем whisper-small...")
        model = whisper.load_model("small")
    
    log_callback("Транскрибирование аудио...")
    log_callback(f"Обработка файла: {audio_path}")
    
    # Оптимизация для CPU - используем fp16=False и beam_size=1 для ускорения
    transcribe_options = {
        'language': 'ru',
        'word_timestamps': True,
        'verbose': False,
        'fp16': False,  # Отключаем fp16 для CPU
        'beam_size': 1,  # Уменьшаем beam size для ускорения
        'best_of': 1,    # Используем только один проход
        'temperature': 0,  # Детерминированный вывод
        'no_speech_threshold': 0.6,  # Повышаем порог для пропуска тишины
        'logprob_threshold': -1.0,   # Ускорение обработки
        'compression_ratio_threshold': 2.4,  # Ускорение обработки
    }
    
    result = model.transcribe(audio_path, **transcribe_options)
    
    audio = AudioSegment.from_file(audio_path)
    log_callback("Анализ лемм запрещенных слов...")
    prohibited_lemmas = find_all_prohibited_lemmas(prohibited_words_list)
    words_to_mute = []
    
    for segment in result['segments']:
        for word_info in segment.get('words', []):
            spoken_word = word_info['word'].strip().lower()
            spoken_lemma = lemmatize_word(spoken_word)
            if spoken_lemma in prohibited_lemmas:
                words_to_mute.append({
                    'word': spoken_word,
                    'start': word_info['start'],
                    'end': word_info['end'],
                    'match_type': 'exact',
                    'base_lemma': spoken_lemma,
                    'similarity': 1.0
                })
            else:
                for lemma in prohibited_lemmas:
                    similarity = jaccard_similarity(spoken_lemma, lemma)
                    if similarity >= similarity_threshold:
                        words_to_mute.append({
                            'word': spoken_word,
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'match_type': 'fuzzy',
                            'base_lemma': lemma,
                            'similarity': similarity
                        })
                        break
    
    words_to_mute.sort(key=lambda x: x['start'], reverse=True)
    if len(words_to_mute) == 1:
        log_callback(f"Найдено {len(words_to_mute)} запрещенное слово для замены...")
    elif len(words_to_mute) >= 2 and len(words_to_mute) <= 4:
        log_callback(f"Найдено {len(words_to_mute)} запрещенных слова для замены...")
    else:
        log_callback(f"Найдено {len(words_to_mute)} запрещенных слов для замены...")
    muted_words = []
    
    for word_data in words_to_mute:
        start_ms = int(word_data['start'] * 1000)
        end_ms = int(word_data['end'] * 1000)
        duration = end_ms - start_ms
        beep = generate_beep(duration_ms=duration)
        audio = audio[:start_ms] + beep + audio[end_ms:]
        match_type = "точное совпадение" if word_data['match_type'] == 'exact' else f"нечеткое совпадение (сходство {word_data['similarity']:.2f})"
        log_callback(f"Заменено слово '{word_data['word']}' — {match_type} с леммой '{word_data['base_lemma']}', время: {start_ms}мс до {end_ms}мс")
        muted_words.append(word_data)
    
    if not words_to_mute:
        log_callback("Запрещенные слова не найдены.")
    
    log_callback(f"Сохранение обработанного файла: {output_path}")
    
    # Перед сохранением проверяем доступ к директории
    try:
        audio.export(output_path, format="wav")
    except Exception as e:
        log_callback(f"Ошибка при сохранении файла: {str(e)}")
        # Пробуем сохранить во временную директорию
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, os.path.basename(output_path))
        log_callback(f"Пробуем сохранить во временную директорию: {temp_file}")
        audio.export(temp_file, format="wav")
        output_path = temp_file
        
    stats = {}
    for item in muted_words:
        base_word = item['base_lemma']
        if base_word not in stats:
            stats[base_word] = []
        stats[base_word].append(item['word'])
    return muted_words, stats


class RecordingThread(threading.Thread):
    def __init__(self, filename, callback=None):
        super().__init__()
        self.filename = os.path.abspath(filename)  # Всегда используем абсолютный путь
        self.callback = callback
        self.is_recording = False
        self.daemon = True
        
    def run(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        
        frames = []
        self.is_recording = True
        
        if self.callback:
            self.callback("Запись начата...")
        
        while self.is_recording:
            data = stream.read(1024)
            frames.append(data)
        
        if self.callback:
            self.callback("Запись завершена.")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Make sure the directory exists before saving
        try:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            
            # Сохранение записи
            with wave.open(self.filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames))
            
            if self.callback:
                self.callback(f"Аудио сохранено в {self.filename}")
        except Exception as e:
            if self.callback:
                self.callback(f"Ошибка при сохранении записи: {str(e)}")
                # Пробуем сохранить во временную директорию
                import tempfile
                temp_dir = tempfile.gettempdir()
                temp_file = os.path.join(temp_dir, os.path.basename(self.filename))
                self.callback(f"Пробуем сохранить во временную директорию: {temp_file}")
                
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(frames))
                
                self.filename = temp_file
                self.callback(f"Аудио сохранено во временную директорию: {self.filename}")
    
    def stop(self):
        self.is_recording = False


class AudioCensorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аудио Цензор")
        self.root.geometry("800x650")
        self.root.minsize(800, 650)
        
        # Проверяем и создаем необходимые директории
        try:
            ensure_directories()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать необходимые директории: {str(e)}")
        
        # Переменные для хранения путей к файлам
        self.current_audio_path = None
        self.output_audio_path = None
        self.recording_thread = None
        
        # Запрещенные слова
        self.prohibited_words = bad_words.get_bad_words()  # По умолчанию
        
        # Переменная для отображения запрещенных слов
        self.show_words = tk.BooleanVar(value=False)
        
        # Создание и настройка интерфейса
        self.create_widgets()
        
        # Переменная для отслеживания прогресса
        self.is_processing = False
        
        # Теперь логируем инициализацию - ПОСЛЕ создания виджетов
        self.log("Инициализация приложения...")
        self.log(f"Рабочая директория: {os.path.abspath(os.curdir)}")
        self.log(f"Директория скрипта: {SCRIPT_DIR}")
        
        # Проверяем наличие кастомной модели
        custom_model_path = get_abs_path("Model/whisper_golos.pt")
        if os.path.exists(custom_model_path):
            self.log(f"Найдена кастомная модель: {custom_model_path}")
        else:
            self.log("Кастомная модель не найдена, будет использована whisper-small")
        
        self.log("Приложение готово к работе. Загрузите или запишите аудиофайл.")
    
    def create_widgets(self):
        # Основной контейнер
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Аудио Цензор", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Фрейм для аудио контролов
        audio_frame = ttk.LabelFrame(main_frame, text="Аудио", padding=10)
        audio_frame.pack(fill=tk.X, pady=10)
        
        # Кнопки для работы с аудио
        btn_frame = ttk.Frame(audio_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.upload_btn = ttk.Button(btn_frame, text="Загрузить аудио", command=self.upload_audio)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.record_btn = ttk.Button(btn_frame, text="Записать аудио", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(btn_frame, text="Воспроизвести", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = ttk.Button(btn_frame, text="Обработать", command=self.process_audio, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Информация о текущем аудиофайле
        self.file_info_label = ttk.Label(audio_frame, text="Аудиофайл не выбран")
        self.file_info_label.pack(fill=tk.X, pady=5)
        
        # Фрейм для настроек цензуры
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки цензуры", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Чекбокс для отображения запрещенных слов
        show_words_frame = ttk.Frame(settings_frame)
        show_words_frame.pack(fill=tk.X, pady=5)
        
        self.show_words_checkbox = ttk.Checkbutton(
            show_words_frame, 
            text="Отображать запрещенные слова", 
            variable=self.show_words,
            command=self.toggle_words_visibility
        )
        self.show_words_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Поле для ввода запрещенных слов
        words_frame = ttk.Frame(settings_frame)
        words_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(words_frame, text="Запрещенные слова (через запятую):").pack(side=tk.LEFT, padx=5)
        
        self.words_entry = ttk.Entry(words_frame)
        self.words_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        masked_words = ["*" * len(word) for word in self.prohibited_words]
        self.words_entry.insert(0, ", ".join(masked_words))
        self.words_entry.config(state=tk.DISABLED)
        
        self.update_words_btn = ttk.Button(words_frame, text="Обновить", command=self.update_prohibited_words)
        self.update_words_btn.pack(side=tk.LEFT, padx=5)
        
        # Настройка порога сходства
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="Порог сходства:").pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=0.75)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=1.0, value=0.75, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.75")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        threshold_scale.config(command=self.update_threshold_label)
        
        # Лог обработки
        log_frame = ttk.LabelFrame(main_frame, text="Журнал обработки", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Текстовое поле для лога
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Полоса прокрутки для лога
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Строка состояния
        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Прогресс-бар
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def toggle_words_visibility(self):
        """Переключает видимость запрещенных слов"""
        # Очищаем поле перед изменением
        self.words_entry.config(state=tk.NORMAL)
        self.words_entry.delete(0, tk.END)
        
        if self.show_words.get():
            # Показываем слова
            self.words_entry.insert(0, ", ".join(self.prohibited_words))
            self.update_words_btn.config(state=tk.NORMAL)
            self.log("Запрещенные слова отображены")
        else:
            # Скрываем слова - показываем звездочки
            masked_words = ["*" * len(word) for word in self.prohibited_words]
            self.words_entry.insert(0, ", ".join(masked_words))
            self.words_entry.config(state=tk.DISABLED)
            self.update_words_btn.config(state=tk.DISABLED)
            self.log("Запрещенные слова скрыты")
    
    def update_threshold_label(self, value):
        self.threshold_label.config(text=f"{float(value):.2f}")
    
    def update_prohibited_words(self):
        if not self.show_words.get():
            return
            
        words_text = self.words_entry.get().strip()
        if words_text:
            self.prohibited_words = [word.strip() for word in words_text.split(",")]
            self.log(f"Список запрещенных слов обновлен: {', '.join(self.prohibited_words)}")
        else:
            self.prohibited_words = []
            self.log("Список запрещенных слов очищен")
    
    def upload_audio(self):
        if self.is_processing:
            return
            
        file_path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Аудиофайлы", "*.wav;*.mp3;*.ogg"), ("Все файлы", "*.*")]
        )
        
        if file_path:
            # Используем абсолютный путь
            file_path = os.path.abspath(file_path)
            filename = os.path.basename(file_path)
            input_dir = get_abs_path(os.path.join("Audio", "Input_audios"))
            
            # Создаем директорию, если она не существует
            try:
                os.makedirs(input_dir, exist_ok=True)
                
                new_path = os.path.join(input_dir, filename)
                
                # Копирование только если файл находится вне целевой директории
                if os.path.abspath(file_path) != os.path.abspath(new_path):
                    import shutil
                    try:
                        shutil.copy2(file_path, new_path)
                        self.log(f"Файл скопирован в: {new_path}")
                    except Exception as e:
                        self.log(f"Ошибка при копировании файла: {str(e)}")
                        # Используем оригинальный путь, если не удалось скопировать
                        new_path = file_path
                        
                self.current_audio_path = new_path
                self.file_info_label.config(text=f"Выбран файл: {filename}")
                self.log(f"Аудиофайл загружен: {filename}")
                self.log(f"Полный путь: {self.current_audio_path}")
                
                # Проверяем, что файл действительно существует
                if not os.path.exists(self.current_audio_path):
                    self.log(f"ОШИБКА: Файл не найден по пути {self.current_audio_path}")
                    messagebox.showerror("Ошибка", f"Не удалось найти файл по пути: {self.current_audio_path}")
                    return
                
                # Разблокировка кнопок
                self.play_btn.config(state=tk.NORMAL)
                self.process_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.log(f"Ошибка при подготовке файла: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось подготовить файл: {str(e)}")
    
    def toggle_recording(self):
        if self.is_processing:
            return
                
        if self.recording_thread and self.recording_thread.is_recording:
            # Остановка записи
            self.record_btn.config(text="Записать аудио")
            self.recording_thread.stop()
        else:
            # Начало записи
            try:
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                
                # Use absolute path for recording file
                input_dir = get_abs_path(os.path.join("Audio", "Input_audios"))
                os.makedirs(input_dir, exist_ok=True)  # Make sure directory exists
                
                self.current_audio_path = os.path.join(input_dir, filename)
                
                self.recording_thread = RecordingThread(self.current_audio_path, callback=self.log)
                self.recording_thread.start()
                
                self.record_btn.config(text="Остановить запись")
                self.file_info_label.config(text=f"Идет запись: {filename}")
                
                # Запускаем таймер проверки записи
                self.root.after(500, self.check_recording)
            except Exception as e:
                self.log(f"Ошибка при начале записи: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось начать запись: {str(e)}")
    
    def check_recording(self):
        if self.recording_thread and not self.recording_thread.is_recording:
            # Запись завершена
            self.record_btn.config(text="Записать аудио")
            filename = os.path.basename(self.current_audio_path)
            self.file_info_label.config(text=f"Выбран файл: {filename}")
            
            # Проверяем существование записанного файла
            if os.path.exists(self.current_audio_path):
                # Разблокировка кнопок
                self.play_btn.config(state=tk.NORMAL)
                self.process_btn.config(state=tk.NORMAL)
            else:
                self.log(f"Предупреждение: записанный файл не найден: {self.current_audio_path}")
                
        elif self.recording_thread and self.recording_thread.is_recording:
            # Продолжаем проверку
            self.root.after(500, self.check_recording)
    
    def play_audio(self):
        if not self.current_audio_path:
            self.log("Ошибка: путь к аудиофайлу не задан")
            return
            
        if not os.path.exists(self.current_audio_path):
            self.log(f"Ошибка: аудиофайл не найден по пути: {self.current_audio_path}")
            return
        
        # Используем системный плеер для воспроизведения
        import platform
        import subprocess
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(self.current_audio_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", self.current_audio_path])
            else:  # Linux
                subprocess.call(["xdg-open", self.current_audio_path])
            
            self.log(f"Воспроизведение: {os.path.basename(self.current_audio_path)}")
        except Exception as e:
            self.log(f"Ошибка при воспроизведении: {str(e)}")
    
    def process_audio(self):
        if self.is_processing:
            return
            
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            messagebox.showerror("Ошибка", "Аудиофайл не выбран или не найден")
            return
        
        if not self.prohibited_words:
            messagebox.showerror("Ошибка", "Список запрещенных слов пуст")
            return
        
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        
        # Подготовка выходного файла
        filename = os.path.basename(self.current_audio_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_censored{ext}"
        
        output_dir = get_abs_path(os.path.join("Audio", "Output_audios"))
        os.makedirs(output_dir, exist_ok=True)
        self.output_audio_path = os.path.join(output_dir, output_filename)
        
        # Запуск обработки в отдельном потоке
        threading.Thread(target=self._process_audio_thread, daemon=True).start()
    
    def _process_audio_thread(self):
        try:
            self.log("Начинается обработка аудио...")
            self.status_var.set("Обработка...")
            self.progress_var.set(0)
            
            # Обновляем прогресс
            self.root.after(100, lambda: self.progress_var.set(10))
            
            # Получаем порог сходства
            threshold = self.threshold_var.get()
            
            # Обработка аудио
            muted_words, stats = mute_prohibited_words(
                self.current_audio_path,
                self.output_audio_path,
                self.prohibited_words,
                similarity_threshold=threshold,
                log_callback=self.log_from_thread
            )
            
            # Обновляем прогресс
            self.root.after(100, lambda: self.progress_var.set(100))
            
            # Выводим статистику
            if muted_words:
                self.root.after(100, lambda: self.log("=== СТАТИСТИКА ОБРАБОТКИ ==="))
                for base_word, found_words in stats.items():
                    unique_words = list(set(found_words))
                    self.root.after(100, lambda bw=base_word, uw=unique_words: 
                        self.log(f"Базовое слово '{bw}': найдено вариантов {unique_words} (всего замен: {len([w for w in muted_words if w['base_lemma'] == bw])})"))
                
                self.root.after(100, lambda: self.log(f"Итого обработано слов: {len(muted_words)}"))
                self.root.after(100, lambda: self.log(f"Обработанный файл сохранен: {self.output_audio_path}"))
                
                # Предлагаем воспроизвести результат
                self.root.after(100, self._offer_play_result)
            else:
                self.root.after(100, lambda: self.log("Запрещенные слова не найдены. Файл не изменен."))
            
        except Exception as e:
            error_msg = f"Ошибка при обработке: {str(e)}"
            self.root.after(100, lambda: self.log(error_msg))
            self.root.after(100, lambda: messagebox.showerror("Ошибка", error_msg))
        finally:
            # Восстанавливаем состояние интерфейса
            self.root.after(100, self._reset_processing_state)
    
    def _offer_play_result(self):
        """Предлагает воспроизвести обработанный файл"""
        if os.path.exists(self.output_audio_path):
            result = messagebox.askyesno("Обработка завершена", 
                                       "Обработка завершена успешно! Воспроизвести обработанный файл?")
            if result:
                try:
                    import platform
                    import subprocess
                    system = platform.system()
                    if system == "Windows":
                        os.startfile(self.output_audio_path)
                    elif system == "Darwin":  # macOS
                        subprocess.call(["open", self.output_audio_path])
                    else:  # Linux
                        subprocess.call(["xdg-open", self.output_audio_path])
                except Exception as e:
                    self.log(f"Ошибка при воспроизведении результата: {str(e)}")
    
    def _reset_processing_state(self):
        """Сбрасывает состояние обработки"""
        self.is_processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.status_var.set("Готово")
        self.progress_var.set(0)
    
    def log_from_thread(self, message):
        """Безопасное логирование из потока"""
        self.root.after(0, lambda: self.log(message))
    
    def log(self, message):
        """Добавляет сообщение в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()


def main():
    """Главная функция приложения"""
    try:
        root = tk.Tk()
        app = AudioCensorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Критическая ошибка при запуске приложения: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()