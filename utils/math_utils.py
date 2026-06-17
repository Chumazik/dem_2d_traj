"""Вспомогательные функции для 2‑D векторных операций."""

import numpy as np

def norm(v: np.ndarray) -> float:
    """Возвращает длину вектора."""
    return float(np.linalg.norm(v))

def unit(v: np.ndarray) -> np.ndarray:
    """Возвращает единичный вектор того же направления."""
    n = norm(v)
    if n == 0:
        return np.zeros_like(v)
    return v / n

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Скалярное произведение."""
    return float(np.dot(a, b))

def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    """Векторное произведение в 2‑D (возвращает скаляр, направленный вдоль оси Z)."""
    return float(a[0] * b[1] - a[1] * b[0])

def rotate(v: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает вектор `v` на угол `angle` (рад)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s],
                    [s,  c]])
    return rot @ v