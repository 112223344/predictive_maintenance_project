# Predictive Maintenance Project

## Описание
Проект по предиктивному обслуживанию оборудования.  
Цель — построить модель бинарной классификации для предсказания отказов машин (целевой признак `Machine failure`: 0 — без отказа, 1 — отказ).

## Датасет
AI4I 2020 Predictive Maintenance Dataset:  
- 10 000 записей  
- 14 признаков:  
  - `UID`, `Product ID`, `Type`  
  - `Air temperature`, `Process temperature`  
  - `Rotational speed`, `Torque`, `Tool wear`  
  - `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`  
Данные загружаются напрямую из UC Irvine ML Repository с помощью пакета `ucimlrepo`.

## Структура репозитория
-	`app.py`: Основной файл приложения.
-	`analysis_and_model.py`: Страница с анализом данных и моделью.
-	`presentation.py`: Страница с презентацией проекта.
-	`requirements.txt`: Файл с зависимостями.
-	`data/`: Папка с данными.
-	`README.md`: Описание проекта.

## Установка и запуск
	# Клонировать репозиторий:
	git clone https://github.com/112223344/predictive_maintenance_project.git
	cd predictive_maintenance_project

	# Создать виртуальное окружение
	python -m venv .venv

	# PowerShell (однократно обойти политику):
	Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
	. .\.venv\Scripts\Activate.ps1

	# или CMD:
	.venv\Scripts\activate.bat

	pip install -r requirements.txt

	streamlit run app.py
	
## Результаты
- LogisticRegression на тесте:  
  - ROC AUC ≈ 0.85  
  - Precision/Recall/F1 (см. в приложении)  
- Баланс классов: отказов примерно 4%

## Видео-демонстрация
https://youtu.be/YkxTtFrfoIQ
