from utils.replay_buffer import ReplayBuffer

# 1. Создаем буфер на 5 элементов (очень маленький, чтобы проверить переполнение)
buffer = ReplayBuffer(capacity=5)
print(f"1. Буфер создан. Размер: {len(buffer)}")

# 2. Заполняем данными
print("\n2. Добавляем данные...")
for i in range(1, 8): # Добавляем 7 элементов (1, 2, ..., 7)
    state = [i, i]       # Фейковый стейт
    action = i           # Фейковое действие
    reward = i * 10      # Фейковая награда
    next_state = [i+1, i+1]
    done = False
    
    print(f"   Добавляем опыт №{i} -> Action {action}")
    buffer.push(state, action, reward, next_state, done)

# 3. Проверяем размер (должен быть 5, а не 7, так как capacity=5)
print(f"\n3. Текущий размер буфера: {len(buffer)} (Ожидалось 5)")
if len(buffer) == 5:
    print("✅ Тест на переполнение пройден! (Старые 1 и 2 удалились)")
else:
    print("❌ Ошибка: буфер не удалил старые элементы")

# 4. Пробуем взять сэмпл (случайную пачку)
batch_size = 3
print(f"\n4. Берем случайный сэмпл из {batch_size} элементов...")
states, actions, rewards, next_s, dones = buffer.sample(batch_size)

print(f"   Достали Actions: {actions.tolist()}")
print(f"   Тип данных Actions: {type(actions)}") # Должен быть torch.Tensor

# 5. Проверка случайности (запустим еще раз)
states2, actions2, _, _, _ = buffer.sample(batch_size)
print(f"   Второй раз достали Actions: {actions2.tolist()}")

if actions.tolist() != actions2.tolist():
    print("✅ Тест на случайность пройден! (Батчи разные)")
else:
    print("⚠️ Батчи совпали (такое бывает редко, либо рандом сломался)")