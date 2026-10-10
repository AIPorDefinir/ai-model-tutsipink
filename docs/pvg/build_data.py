import json
import re
import subprocess
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent  # docs/pvg: fuentes y salida de la página
OWNER = "AIPorDefinir"
REPO = f"{OWNER}/ai-model-tutsipink"
PROJECT = "1"
PLACEHOLDER = "Tu actividad aquí"
# Usuario de GitHub -> nombre del integrante (README)
TEAM = {
    "Facundo-Barbera": "Facundo Bautista Barbera",
    "taqueritospro": "Oswaldo Isaias Hernandez Santes",
    "AlejandroSH1": "Alfredo Alejandro Soto Herrera",
    "Emiliano1410": "Emiliano Camacho Ponce",
    "ikerMJHDZ09": "Iker Mejia Hernandez",
    "JorgeManuelOyoqui": "Jorge Manuel Oyoqui Aguilera",
}


def parse_wbs(text):
    project = phase = activity = None
    items = []
    for line in text.splitlines():
        m = re.match(r"(\*+)\s+(\S+)\s+(.*)", line)
        if not m:
            continue
        level, code, name = len(m[1]), m[2].rstrip("."), m[3].strip()
        if level == 1:
            project = f"{m[2]} {name}"
        elif level == 2:
            phase = f"{code}. {name}"
        elif level == 3:
            activity = (code, name)
        elif level == 4:
            items.append({
                "codigo": code,
                "nombre": activity[1] if name == PLACEHOLDER else name,
                "fase": phase,
                "actividad": f"{activity[0]} {activity[1]}",
                "definido": name != PLACEHOLDER,
            })
    return project, items


def fetch_issues():
    out = subprocess.run(
        ["gh", "issue", "list", "-R", REPO, "--state", "all", "--limit", "500",
         "--json", "number,title,state,body,url,assignees"],
        capture_output=True, text=True, check=True,
    ).stdout
    issues = json.loads(out)
    # Se relaciona por el código al inicio del título ("1.5.1 ..."); por nombre si el código no coincide
    by_code = {i["title"].split(" ", 1)[0]: i for i in issues}
    by_name = {i["title"].split(" ", 1)[-1].strip(): i for i in issues}
    return by_code, by_name


def progress(issue):
    if issue["state"] == "CLOSED":
        return 1.0
    boxes = re.findall(r"- \[([ xX])\]", issue["body"])
    return sum(b != " " for b in boxes) / len(boxes) if boxes else 0.0


def fetch_actual_hours():
    out = subprocess.run(
        ["gh", "project", "item-list", PROJECT, "--owner", OWNER, "--format", "json", "--limit", "500"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {
        i["content"]["number"]: float(i.get("horas reales") or 0)
        for i in json.loads(out)["items"]
        if i["content"].get("type") == "Issue"
    }


def main():
    project, wbs = parse_wbs((HERE / "crisp-dm-wbs.puml").read_text())
    plan = json.loads((HERE / "plan.json").read_text())
    by_code, by_name = fetch_issues()
    hours = fetch_actual_hours()
    tasks = []
    for item in wbs:
        if item["codigo"] not in plan:
            print(f"Sin planificar en plan.json: {item['codigo']} {item['nombre']}")
            continue
        issue = by_code.get(item["codigo"]) or by_name.get(item["nombre"])
        tasks.append({
            **item,
            **plan[item["codigo"]],
            "issue": issue["number"] if issue else None,
            "url": issue["url"] if issue else None,
            "responsables": [TEAM.get(a["login"], a["login"]) for a in issue["assignees"]] if issue else [],
            "avance": round(progress(issue), 3) if issue else 0.0,
            "horas_reales": hours.get(issue["number"], 0.0) if issue else 0.0,
        })
    data = {"repo": REPO, "proyecto": project, "corte": date.today().isoformat(), "tareas": tasks}
    out = HERE / "data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    linked = sum(t["issue"] is not None for t in tasks)
    print(f"{len(tasks)} entregables ({linked} con issue) -> {out}")


if __name__ == "__main__":
    main()
