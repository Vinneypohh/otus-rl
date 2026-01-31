import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np

def manual_car_control():
    print("Запускаем CarRacing-v3!")
    print("=== УПРАВЛЕНИЕ ===")
    print("  [W] - Газ")
    print("  [S] - Тормоз")
    print("  [A] - Руль влево")
    print("  [D] - Руль вправо")
    print("  [Пробел] - Резкий тормоз (Ручник)")
    
    # Создаем среду
    # render_mode="rgb_array" обязателен для утилиты play
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Определяем действия (dtype=np.float32 обязателен!)
    # Формат: [steering, gas, brake]
    
    # Основные кнопки
    gas    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    brake  = np.array([0.0, 0.0, 0.8], dtype=np.float32) # не полный тормоз, чтобы не юзило
    left   = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    right  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Комбинации (чтобы можно было газовать и рулить одновременно)
    gas_left  = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
    gas_right = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    
    mapping = {
        (ord('w'),): gas,
        (ord('s'),): brake,
        (ord('a'),): left,
        (ord('d'),): right,
        (ord(' '),): np.array([0.0, 0.0, 1.0], dtype=np.float32), # Пробел - стоп-кран
        
        # Комбинации газ + поворот
        (ord('w'), ord('a')): gas_left,
        (ord('w'), ord('d')): gas_right,
    }
    
    # noop = ничего не делать (руль прямо, газ 0, тормоз 0)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # zoom=2 делает окно побольше, так играть приятнее
    play(env, keys_to_action=mapping, noop=noop, fps=30, zoom=2)

if __name__ == "__main__":
    manual_car_control()
