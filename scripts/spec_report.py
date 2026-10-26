"""Corre los tests de positioner/ y muestra los specs con cada escenario tildado.

Lee los spec.md de upgrade/specs/, busca en cada escenario la línea
`- **TEST** nombre_del_test` (con el nombre entre backticks), corre esos tests con unittest y reimprime el
spec al estilo Cucumber: requirement, escenario con ✔ o ✘, y sus pasos.

Estados de un escenario:
    ✔ pasó          el test pasó
    ✘ falló         el test falló, dio error o no existe en la suite
    ↷ salteado      el test está marcado con skip
    ? sin test      el escenario no tiene línea TEST (spec escrito antes que el test)

Sale con código 1 si algún escenario falló. Con --estricto, también si alguno
quedó sin test.

Uso (desde la raíz del repo):
    python scripts/spec_report.py
    python scripts/spec_report.py upgrade/specs/merchant-feed/spec.md
    python scripts/spec_report.py --estricto
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SPECS = RAIZ / "upgrade" / "specs"
TESTS = RAIZ / "positioner" / "tests"

RE_TEST = re.compile(r"^- \*\*TEST\*\* `(\w+)`")
RE_PASO = re.compile(r"^- \*\*(\w+)\*\* (.*)")


# ── Lectura de los spec.md ────────────────────────────────────────────────────

@dataclass
class Escenario:
    nombre: str
    pasos: list[str] = field(default_factory=list)
    test: str | None = None


@dataclass
class Requirement:
    nombre: str
    escenarios: list[Escenario] = field(default_factory=list)


@dataclass
class Spec:
    ruta: Path
    titulo: str
    requirements: list[Requirement] = field(default_factory=list)


def leer_spec(ruta: Path) -> Spec:
    spec = Spec(ruta=ruta, titulo=ruta.parent.name)
    requirement = None
    escenario = None
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if linea.startswith("# "):
            spec.titulo = linea[2:].strip()
        elif linea.startswith("### Requirement:"):
            requirement = Requirement(linea.split(":", 1)[1].strip())
            spec.requirements.append(requirement)
            escenario = None
        elif linea.startswith("#### Scenario:") and requirement is not None:
            escenario = Escenario(linea.split(":", 1)[1].strip())
            requirement.escenarios.append(escenario)
        elif linea.startswith("#"):
            # Cualquier otro encabezado (## Tests, ## ADDED Requirements) cierra el bloque.
            requirement = None
            escenario = None
        elif escenario is not None:
            if m := RE_TEST.match(linea):
                escenario.test = m.group(1)
            elif m := RE_PASO.match(linea):
                escenario.pasos.append(f"{m.group(1)} {m.group(2)}")
            elif linea.startswith("  ") and escenario.pasos:
                # Continuación de un paso partido en varios renglones.
                escenario.pasos[-1] += " " + linea.strip()
    return spec


# ── Ejecución de los tests ────────────────────────────────────────────────────

def aplanar(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from aplanar(item)
        else:
            yield item


class Resultados(unittest.TestResult):
    """Guarda el estado de cada test por nombre de método."""

    def __init__(self) -> None:
        super().__init__()
        self.estados: dict[str, tuple[str, str]] = {}

    def addSuccess(self, test):
        self.estados[test._testMethodName] = ("pasó", "")

    def addFailure(self, test, err):
        self.estados[test._testMethodName] = ("falló", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        self.estados[test._testMethodName] = ("falló", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        self.estados[test._testMethodName] = ("salteado", reason)


def correr_tests(nombres: set[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Corre los tests pedidos. Devuelve sus estados y los errores de importación."""
    sys.path.insert(0, str(RAIZ))
    descubiertos = unittest.defaultTestLoader.discover(str(TESTS), top_level_dir=str(RAIZ))

    por_nombre: dict[str, unittest.TestCase] = {}
    errores_carga = []
    for test in aplanar(descubiertos):
        if isinstance(test, unittest.loader._FailedTest):
            errores_carga.append(f"{test.id()}: {test._exception}")
            continue
        nombre = test._testMethodName
        if nombre in por_nombre:
            errores_carga.append(f"{nombre} está repetido: {por_nombre[nombre].id()} y {test.id()}")
        por_nombre[nombre] = test

    resultados = Resultados()
    unittest.TestSuite(por_nombre[n] for n in sorted(nombres) if n in por_nombre).run(resultados)
    for nombre in nombres - por_nombre.keys():
        resultados.estados[nombre] = ("falló", f"No hay ningún test llamado {nombre} en {TESTS.relative_to(RAIZ)}")
    return resultados.estados, errores_carga


# ── Reporte ───────────────────────────────────────────────────────────────────

COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
SIMBOLOS = {"pasó": ("✔", "32"), "falló": ("✘", "31"), "salteado": ("↷", "33"), "sin test": ("?", "33")}
PLURALES = {"pasó": "pasaron", "falló": "fallaron", "salteado": "salteados", "sin test": "sin test"}


def pintar(texto: str, codigo: str) -> str:
    return f"\033[{codigo}m{texto}\033[0m" if COLOR else texto


def resumir(detalle: str) -> str:
    """Devuelve la línea de la excepción: la primera sin sangría después del último frame."""
    lineas = detalle.strip().splitlines()
    frames = [i for i, linea in enumerate(lineas) if linea.startswith("  ")]
    return lineas[frames[-1] + 1] if frames and frames[-1] + 1 < len(lineas) else lineas[0]


def mostrar_ruta(ruta: Path) -> Path:
    return ruta.relative_to(RAIZ) if ruta.is_relative_to(RAIZ) else ruta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("specs", nargs="*", type=Path, help="spec.md a reportar (por defecto, todos los de upgrade/specs/)")
    parser.add_argument("--estricto", action="store_true", help="falla también si hay escenarios sin test")
    parser.add_argument("--sin-pasos", action="store_true", help="no muestra los pasos WHEN/THEN de cada escenario")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    rutas = args.specs or sorted(SPECS.glob("*/spec.md"))
    specs = sorted((leer_spec(r.resolve()) for r in rutas), key=lambda s: s.titulo)
    nombres = {e.test for s in specs for r in s.requirements for e in r.escenarios if e.test}
    estados, errores_carga = correr_tests(nombres)

    totales = {estado: 0 for estado in SIMBOLOS}
    detalles = []
    for spec in specs:
        print(pintar(spec.titulo, "1"), pintar(f"  {mostrar_ruta(spec.ruta)}", "2"))
        for requirement in spec.requirements:
            print(f"  Requirement: {requirement.nombre}")
            for escenario in requirement.escenarios:
                estado, detalle = estados[escenario.test] if escenario.test else ("sin test", "")
                totales[estado] += 1
                simbolo, codigo = SIMBOLOS[estado]
                print(f"    {pintar(simbolo, codigo)} Scenario: {escenario.nombre}")
                if not args.sin_pasos:
                    for paso in escenario.pasos:
                        print(pintar(f"        {paso}", "2"))
                if estado == "falló":
                    print(pintar(f"        {resumir(detalle)}", codigo))
                    detalles.append((spec.titulo, escenario, detalle))
        print()

    for error in errores_carga:
        print(pintar(f"Error al cargar los tests: {error}", "31"))
    for titulo, escenario, detalle in detalles:
        print(pintar(f"── {titulo} › {escenario.nombre} ({escenario.test})", "31"))
        print(detalle)

    total = sum(totales.values())
    partes = [pintar(f"{n} {estado if n == 1 else PLURALES[estado]}", SIMBOLOS[estado][1]) for estado, n in totales.items() if n]
    print(f"{total} escenarios: " + ", ".join(partes))

    hay_fallas = totales["falló"] or errores_carga or (args.estricto and totales["sin test"])
    return 1 if hay_fallas else 0


if __name__ == "__main__":
    sys.exit(main())
