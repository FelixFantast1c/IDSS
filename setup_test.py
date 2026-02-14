import os
import sys

# Вимикаємо системні логи TensorFlow перед імпортом
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def check_component(name, import_name=None):
    if import_name is None:
        import_name = name
    
    try:
        if name == 'Python':
            import platform
            print(f"{name:<15} : OK (v{platform.python_version()})")
            return

        lib = __import__(import_name)
        version = getattr(lib, '__version__', 'Unknown')
        status = f"OK (v{version})"

        # Специфічна перевірка для TensorFlow та GPU
        if name == 'TensorFlow':
            import tensorflow as tf
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    gpu_names = ", ".join([tf.config.experimental.get_device_details(g).get('device_name', 'GPU') for g in gpus])
                    status += f" | GPU ENABLED: {len(gpus)} ({gpu_names})"
                else:
                    status += " | WARNING: CPU ONLY (No GPU detected)"
            except Exception as e:
                status += f" | GPU Check Failed: {e}"

        # Перевірка Keras бекенду
        if name == 'Keras':
            import keras
            backend = keras.backend.backend()
            status += f" | Backend: {backend}"

        print(f"{name:<15} : {status}")

    except ImportError:
        print(f"{name:<15} : FAILED (Not Installed)")
    except Exception as e:
        print(f"{name:<15} : ERROR ({e})")

if __name__ == "__main__":
    print("-" * 85)
    print(f"{'COMPONENT':<15} : STATUS")
    print("-" * 85)

    check_component("Python")
    check_component("TensorFlow", "tensorflow")
    check_component("Keras", "keras")
    check_component("NumPy", "numpy")
    check_component("Pandas", "pandas")
    check_component("Scikit-learn", "sklearn")
    check_component("Matplotlib", "matplotlib")
    check_component("Seaborn", "seaborn")
    check_component("Keras Tuner", "keras_tuner")
    
    print("-" * 85)