import polars as pl
import json
import sys
import io
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import traceback

console = Console()


class FHIRParser:
    def __init__(self, resources):
        self.resources = resources

    def get_by_type(self, res_type):
        return [r for r in self.resources if r.get("resourceType") == res_type]

    def parse_observations(self):
        obs_list = self.get_by_type("Observation")
        if not obs_list:
            return None

        data = []
        for r in obs_list:
            row = {
                "时间": r.get("effectiveDateTime", "-")[:16],
                "项目": r.get("code", {}).get("text", "未知"),
                "数值": None,
                "单位": "",
            }
            # 1. 处理标准数值
            if "valueQuantity" in r:
                row["数值"] = r["valueQuantity"].get("value")
                row["单位"] = r["valueQuantity"].get("unit", "")
            # 2. 处理血压等复合组件
            elif "component" in r:
                vals = []
                for comp in r["component"]:
                    v = comp.get("valueQuantity", {}).get("value")
                    if v is not None:
                        vals.append(str(v))
                row["数值"] = "/".join(vals) if vals else "复合项"
                row["单位"] = r["component"][0].get("valueQuantity", {}).get("unit", "")
            # 3. 处理分类/枚举值 (例如：阳性/阴性)
            elif "valueCodeableConcept" in r:
                row["数值"] = r["valueCodeableConcept"].get("text", "-")

            data.append(row)
        return pl.DataFrame(data).sort("时间", descending=True)

    def parse_reports(self):
        # 诊断报告通常包含医生最终的定性描述
        reports = self.get_by_type("DiagnosticReport")
        if not reports:
            return None
        return pl.DataFrame(
            [
                {
                    "日期": r.get("effectiveDateTime", "-")[:10],
                    "类别": r.get("category", [{}])[0]
                    .get("coding", [{}])[0]
                    .get("display", "检查"),
                    "报告名称": r.get("code", {}).get("text", "未知"),
                    "结论": r.get("conclusion", "详见细节"),
                }
                for r in reports
            ]
        ).sort("日期", descending=True)

    def parse_medications(self):
        med_list = self.get_by_type("MedicationRequest")
        if not med_list:
            return None
        return pl.DataFrame(
            [
                {
                    "日期": m.get("authoredOn", "-")[:10],
                    "药物名称": m.get("medicationCodeableConcept", {}).get(
                        "text", "未知"
                    ),
                    "状态": m.get("status"),
                    "剂量": m.get("dosageInstruction", [{}])[0].get("text", "-"),
                }
                for m in med_list
            ]
        ).sort("日期", descending=True)

    def parse_procedures(self):
        proc_list = self.get_by_type("Procedure")
        if not proc_list:
            return None
        return pl.DataFrame(
            [
                {
                    "执行时间": p.get("performedPeriod", {}).get(
                        "start", p.get("performedDateTime", "-")
                    )[:16],
                    "操作名称": p.get("code", {}).get("text", "未知"),
                    "状态": p.get("status"),
                }
                for p in proc_list
            ]
        ).sort("执行时间", descending=True)

    def parse_conditions(self):
        conds = self.get_by_type("Condition")
        if not conds:
            return None
        rows = []
        for c in conds:
            onset = (
                c.get("onsetDateTime")
                or c.get("onset")
                or c.get("assertedDate")
                or c.get("recordedDate")
                or "-"
            )
            code = (
                c.get("code", {}).get("text")
                or c.get("code", {}).get("coding", [{}])[0].get("display")
                or c.get("code", {}).get("coding", [{}])[0].get("code")
                or "未知"
            )

            # clinicalStatus 或 verificationStatus，优先显示可读文本或 coding.display/code
            clinical = "-"
            cs = c.get("clinicalStatus") or c.get("verificationStatus")
            if isinstance(cs, dict):
                clinical = (
                    cs.get("text")
                    or cs.get("coding", [{}])[0].get("display")
                    or cs.get("coding", [{}])[0].get("code")
                    or str(cs)
                )
            else:
                clinical = cs or "-"

            # 严重程度也可能是 dict
            sev_val = c.get("severity")
            if isinstance(sev_val, dict):
                severity = (
                    sev_val.get("text")
                    or sev_val.get("coding", [{}])[0].get("display")
                    or sev_val.get("coding", [{}])[0].get("code")
                    or "-"
                )
            else:
                severity = sev_val or "-"

            rows.append(
                {
                    "发现时间": onset[:10],
                    "诊断": code,
                    "阶段/状态": clinical,
                    "严重程度": severity,
                }
            )
        return pl.DataFrame(rows).sort("发现时间", descending=True)

    def parse_allergies(self):
        al = self.get_by_type("AllergyIntolerance")
        if not al:
            return None
        rows = []
        for a in al:
            when = a.get("recordedDate") or a.get("onset") or "-"
            code = a.get("code", {}).get("text", "未知")
            crit = a.get("criticality", "-")
            reactions = []
            for r in a.get("reaction", []):
                mans = [
                    m.get("text", "")
                    for m in r.get("manifestation", [])
                    if m.get("text")
                ]
                if mans:
                    reactions.extend(mans)
            reactions = ", ".join(reactions) if reactions else "-"
            rows.append(
                {
                    "发现时间": when[:10],
                    "过敏源": code,
                    "反应": reactions,
                    "严重性": crit,
                }
            )
        return pl.DataFrame(rows).sort("发现时间", descending=True)

    def parse_immunizations(self):
        imms = self.get_by_type("Immunization")
        if not imms:
            return None
        rows = []
        for i in imms:
            date = (
                i.get("occurrenceDateTime")
                or i.get("occurrence")
                or i.get("date")
                or "-"
            )
            vac = i.get("vaccineCode", {}).get("text", "未知")
            status = i.get("status", "-")
            lot = i.get("lotNumber", "-")
            rows.append(
                {"接种时间": date[:10], "疫苗": vac, "状态": status, "批号": lot}
            )
        return pl.DataFrame(rows).sort("接种时间", descending=True)

    def parse_encounters(self):
        encs = self.get_by_type("Encounter")
        if not encs:
            return None
        rows = []
        for e in encs:
            start = e.get("period", {}).get("start") or e.get("start") or "-"
            end = e.get("period", {}).get("end") or e.get("end") or "-"
            typ = (
                e.get("type", [{}])[0].get("text")
                if e.get("type")
                else e.get("class", {}).get("display", "-")
            )
            status = e.get("status", "-")
            rows.append(
                {
                    "开始": start[:16],
                    "结束": end[:16] if end else "-",
                    "类型": typ or "-",
                    "状态": status,
                }
            )
        return pl.DataFrame(rows).sort("开始", descending=True)

    def parse_careplans(self):
        cps = self.get_by_type("CarePlan")
        if not cps:
            return None
        rows = []
        for c in cps:
            start = c.get("period", {}).get("start") or c.get("created") or "-"
            status = c.get("status", "-")
            title = c.get("title") or c.get("description") or c.get("intent") or "-"
            rows.append({"起始": start[:10], "状态": status, "描述": title})
        return pl.DataFrame(rows).sort("起始", descending=True)

    def parse_careteam(self):
        teams = self.get_by_type("CareTeam")
        if not teams:
            return None
        rows = []
        for t in teams:
            period = t.get("period", {})
            start = period.get("start") or "-"
            end = period.get("end") or "-"
            name = t.get("name") or t.get("category", [{}])[0].get("text") or "-"
            members = len(t.get("participant", []))
            status = t.get("status", "-")
            rows.append({"名称": name, "开始": start[:10], "结束": end[:10] if end else "-", "成员数": members, "状态": status})
        return pl.DataFrame(rows).sort("开始", descending=True)

    def parse_claims(self):
        claims = self.get_by_type("Claim")
        if not claims:
            return None
        rows = []
        for c in claims:
            created = c.get("created") or c.get("date") or "-"
            status = c.get("status", "-")
            use = c.get("use", "-")
            ctype = c.get("type", {}).get("text") or (c.get("type", {}).get("coding", [{}])[0].get("display") if c.get("type") else "-")
            total = "-"
            if isinstance(c.get("total"), dict):
                total = c.get("total", {}).get("value") or c.get("total", {}).get("amount", {}).get("value") or "-"
            else:
                total = c.get("total") or "-"
            rows.append({"创建": created[:10], "状态": status, "用途": use, "类型": ctype or "-", "总额": total})
        return pl.DataFrame(rows).sort("创建", descending=True)

    def parse_documentreferences(self):
        docs = self.get_by_type("DocumentReference")
        if not docs:
            return None
        rows = []
        for d in docs:
            date = d.get("created") or d.get("indexed") or d.get("date") or "-"
            title = d.get("description") or d.get("title") or (d.get("type", {}).get("text") if d.get("type") else "-")
            status = d.get("status", "-")
            formats = ",".join([c.get("format", {}).get("display", c.get("format", {}).get("code", "")) or "" for c in d.get("content", [])])
            attachments = len(d.get("content", []))
            rows.append({"日期": date[:10], "标题": title, "状态": status, "格式": formats or "-", "附件数": attachments})
        return pl.DataFrame(rows).sort("日期", descending=True)

    def parse_explanationofbenefits(self):
        eobs = self.get_by_type("ExplanationOfBenefit")
        if not eobs:
            return None
        rows = []
        for e in eobs:
            created = e.get("created") or e.get("date") or "-"
            status = e.get("status", "-")
            provider = e.get("provider", {}).get("display") if isinstance(e.get("provider"), dict) else e.get("provider")
            total = "-"
            # total may be a list of monetary amounts
            if isinstance(e.get("total"), list) and e.get("total"):
                amt = e.get("total")[0].get("amount", {})
                total = f"{amt.get('value', '-')}{amt.get('currency','')}"
            elif isinstance(e.get("total"), dict):
                amt = e.get("total").get("amount", {})
                total = f"{amt.get('value', '-')}{amt.get('currency','')}"
            rows.append({"创建": created[:10], "状态": status, "提供者": provider or "-", "总额": total})
        return pl.DataFrame(rows).sort("创建", descending=True)

    def parse_imagingstudies(self):
        ims = self.get_by_type("ImagingStudy")
        if not ims:
            return None
        rows = []
        for im in ims:
            started = im.get("started") or im.get("startedDateTime") or "-"
            series_count = im.get("numberOfSeries") or len(im.get("series", []))
            instance_count = im.get("numberOfInstances") or sum([s.get("numberOfInstances", 0) for s in im.get("series", [])])
            desc = im.get("description") or im.get("note", [{}])[0].get("text", "-")
            rows.append({"开始": started[:16], "系列数": series_count, "实例数": instance_count, "描述": desc})
        return pl.DataFrame(rows).sort("开始", descending=True)

    def parse_provenances(self):
        provs = self.get_by_type("Provenance")
        if not provs:
            return None
        rows = []
        for p in provs:
            recorded = p.get("recorded") or p.get("occurred") or "-"
            activity = p.get("activity", {}).get("display") if isinstance(p.get("activity"), dict) else p.get("activity")
            agents = []
            for a in p.get("agent", []):
                who = a.get("who", {})
                if isinstance(who, dict):
                    agents.append(who.get("display") or who.get("reference") or "-")
                else:
                    agents.append(str(who))
            rows.append({"时间": recorded[:16], "活动": activity or "-", "代理": ", ".join(agents) or "-"})
        return pl.DataFrame(rows).sort("时间", descending=True)

    def parse_practitioners(self):
        practitioners = self.get_by_type("Practitioner")
        if not practitioners:
            return None
        rows = []
        for p in practitioners:
            pid = p.get("id", "-")
            name_obj = p.get("name", [{}])[0]
            family = name_obj.get("family", "")
            given = " ".join(name_obj.get("given", []))
            name = (family + " " + given).strip() if (family or given) else (name_obj.get("text") or "-")
            active = p.get("active", "-")
            gender = p.get("gender", "-")
            identifiers = ", ".join([i.get("value", "") for i in p.get("identifier", []) if i.get("value")]) or "-"
            telecoms = []
            for t in p.get("telecom", []):
                val = t.get("value")
                system = t.get("system")
                if val:
                    telecoms.append(f"{system or 'tel'}:{val}")
            telecom = ", ".join(telecoms) if telecoms else "-"
            addr = p.get("address", [])
            if addr:
                a = addr[0]
                line = " ".join(a.get("line", []))
                city = a.get("city", "")
                country = a.get("country", "")
                address = ", ".join([s for s in [line, city, country] if s]) or "-"
            else:
                address = "-"
            rows.append({"id": pid, "姓名": name, "活跃": str(active), "性别": gender, "标识": identifiers, "联系方式": telecom, "地址": address})
        return pl.DataFrame(rows).sort("姓名", descending=False)

    def parse_practitioner_roles(self):
        roles = self.get_by_type("PractitionerRole")
        if not roles:
            return None
        rows = []
        for r in roles:
            rid = r.get("id", "-")
            prac = r.get("practitioner")
            practitioner = (
                prac.get("display") if isinstance(prac, dict) else (str(prac) if prac else "-")
            )
            org = r.get("organization")
            organization = (
                org.get("display") if isinstance(org, dict) else (str(org) if org else "-")
            )
            codes = []
            for c in r.get("code", []):
                txt = c.get("text") or (c.get("coding", [{}])[0].get("display") if c.get("coding") else None)
                if txt:
                    codes.append(txt)
            code_text = ", ".join(codes) if codes else "-"
            specs = []
            for s in r.get("specialty", []):
                t = s.get("text") or (s.get("coding", [{}])[0].get("display") if s.get("coding") else None)
                if t:
                    specs.append(t)
            specialty = ", ".join(specs) if specs else "-"
            locations = []
            for loc in r.get("location", []):
                if isinstance(loc, dict):
                    locations.append(loc.get("display") or loc.get("reference") or "-")
                else:
                    locations.append(str(loc))
            locs = ", ".join(locations) if locations else "-"
            telecoms = []
            for t in r.get("telecom", []):
                val = t.get("value")
                system = t.get("system")
                if val:
                    telecoms.append(f"{system or 'tel'}:{val}")
            telecom = ", ".join(telecoms) if telecoms else "-"
            rows.append({"id": rid, "从业者": practitioner, "机构": organization, "职能/代码": code_text, "专长": specialty, "地点": locs, "联系方式": telecom})
        return pl.DataFrame(rows).sort("机构", descending=False)

    def parse_organizations(self):
        orgs = self.get_by_type("Organization")
        if not orgs:
            return None
        rows = []
        for o in orgs:
            oid = o.get("id", "-")
            name = o.get("name") or (o.get("type", [{}])[0].get("text") if o.get("type") else "-")
            active = o.get("active", "-")
            identifiers = ", ".join([i.get("value", "") for i in o.get("identifier", []) if i.get("value")]) or "-"
            telecoms = []
            for t in o.get("telecom", []):
                val = t.get("value")
                system = t.get("system")
                if val:
                    telecoms.append(f"{system or 'tel'}:{val}")
            telecom = ", ".join(telecoms) if telecoms else "-"
            addr = o.get("address", [])
            if addr:
                a = addr[0]
                line = " ".join(a.get("line", []))
                city = a.get("city", "")
                country = a.get("country", "")
                address = ", ".join([s for s in [line, city, country] if s]) or "-"
            else:
                address = "-"
            rows.append({"id": oid, "名称": name, "活跃": str(active), "标识": identifiers, "联系方式": telecom, "地址": address})
        return pl.DataFrame(rows).sort("名称", descending=False)

    def parse_locations(self):
        locs = self.get_by_type("Location")
        if not locs:
            return None
        rows = []
        for l in locs:  # noqa: E741
            lid = l.get("id", "-")
            name = l.get("name") or "-"
            status = l.get("status", "-")
            managing = l.get("managingOrganization")
            managing_org = (
                managing.get("display") if isinstance(managing, dict) else (str(managing) if managing else "-")
            )
            addr = l.get("address", [])
            a = None
            if isinstance(addr, list) and addr:
                a = addr[0]
            elif isinstance(addr, dict):
                a = addr
            if a:
                line = " ".join(a.get("line", [])) if isinstance(a.get("line", []), list) else (a.get("line") or "")
                city = a.get("city", "")
                country = a.get("country", "")
                address = ", ".join([s for s in [line, city, country] if s]) or "-"
            else:
                address = "-"
            position = l.get("position")
            if isinstance(position, dict):
                lat = position.get("latitude")
                lon = position.get("longitude")
                coord = f"{lat},{lon}" if lat is not None and lon is not None else "-"
            else:
                coord = "-"
            telecoms = []
            for t in l.get("telecom", []):
                val = t.get("value")
                system = t.get("system")
                if val:
                    telecoms.append(f"{system or 'tel'}:{val}")
            telecom = ", ".join(telecoms) if telecoms else "-"
            rows.append({"id": lid, "名称": name, "状态": status, "机构": managing_org, "地址": address, "坐标": coord, "联系方式": telecom})
        return pl.DataFrame(rows).sort("名称", descending=False)


def render_table(title, df, color="blue"):
    if df is None or df.is_empty():
        return
    count = len(df)
    title_text = f"{title} — 已解析: {count}"
    table = Table(
        title=f"[{color} bold]{title_text}[/{color} bold]",
        header_style=f"bold white on {color}",
        box=None,
    )
    for col in df.columns:
        table.add_column(col)

    # 只展示前 20 条记录，增加可读性
    for row in df.head(20).rows():
        table.add_row(*[str(item) if item is not None else "-" for item in row])
    console.print(table)
    if len(df) > 20:
        console.print(f"[dim]... 共 {len(df)} 条记录，已自动截断展示[/dim]")
    console.print("")


def summarize_fhir_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        resources = [
            e.get("resource") for e in bundle.get("entry", []) if e.get("resource")
        ]
    except Exception as e:
        console.print(f"[red]读取失败: {e}[/red]")
        return

    parser = FHIRParser(resources)

    # 统计各资源类型数量
    counts_by_type = {}
    for r in resources:
        t = r.get("resourceType", "Unknown")
        counts_by_type[t] = counts_by_type.get(t, 0) + 1
    resource_parsers = {
        "Observation": parser.parse_observations,
        "DiagnosticReport": parser.parse_reports,
        "MedicationRequest": parser.parse_medications,
        "Procedure": parser.parse_procedures,
        "AllergyIntolerance": parser.parse_allergies,
        "Condition": parser.parse_conditions,
        "Immunization": parser.parse_immunizations,
        "Encounter": parser.parse_encounters,
        "CarePlan": parser.parse_careplans,
        "CareTeam": parser.parse_careteam,
        "Claim": parser.parse_claims,
        "DocumentReference": parser.parse_documentreferences,
        "ExplanationOfBenefit": parser.parse_explanationofbenefits,
        "ImagingStudy": parser.parse_imagingstudies,
        "Provenance": parser.parse_provenances,
        "Organization": parser.parse_organizations,
        "Location": parser.parse_locations,
        "Practitioner": parser.parse_practitioners,
        "PractitionerRole": parser.parse_practitioner_roles,
    }

    # 1. 打印头部面板（包含模块组成摘要）
    pat_list = parser.get_by_type("Patient")
    header_info = f"文件路径: {file_path}\n资源总数: {len(resources)}"
    if pat_list:
        p = pat_list[0]
        name = f"{p.get('name',[{}])[0].get('family','')} {' '.join(p.get('name',[{}])[0].get('given',[]))}"
        header_info += f"\n患者信息: [bold yellow]{name}[/bold yellow] | [bold]{p.get('gender')}[/bold] | 生日: {p.get('birthDate')}"

    # 模块组成摘要 lines
    summary_lines = []
    for rt, fn in resource_parsers.items():
        total = counts_by_type.get(rt, 0)
        if total == 0:
            continue
        try:
            df = fn()
            parsed = len(df) if df is not None else 0
        except Exception:
            parsed = 0
        display_name = rt
        summary_lines.append(f"{display_name}: {parsed}/{total}")

    # 未解析类型（不在 resource_parsers 中）
    unhandled_types = [
        t
        for t in sorted(counts_by_type.keys())
        if t not in resource_parsers and t != "Patient"
    ]
    unhandled_summary = (
        ", ".join([f"{t}({counts_by_type.get(t)})" for t in unhandled_types]) or "无"
    )

    # 未解析资源示例（每种类型至多展示一条，最多 5 条）
    examples = []
    for r in resources:
        t = r.get("resourceType")
        if t in unhandled_types and len(examples) < 5:
            rid = r.get("id", "-")
            keys = list(r.keys())
            examples.append(f"{t} id={rid} keys={keys}")

    header_info += "\n\n模块组成:\n" + "\n".join(summary_lines)
    header_info += f"\n\n未解析类型: {unhandled_summary}"
    if examples:
        header_info += "\n示例: " + "; ".join(examples)

    console.print(
        Panel(
            header_info,
            title="[bold white]FHIR Bundle 摘要[/bold white]",
            border_style="cyan",
        )
    )

    # 2. 依次渲染各个分析模块（仅渲染文件中存在的资源类型）
    render_order = [
        ("Observation", "🧬 核心健康监测 (Observation)", "blue"),
        ("DiagnosticReport", "📝 诊断报告结论 (DiagnosticReport)", "cyan"),
        ("MedicationRequest", "💊 处方药物历史 (MedicationRequest)", "magenta"),
        ("Procedure", "🏥 医疗处置/手术 (Procedure)", "red"),
        ("AllergyIntolerance", "⚠️ 过敏/不耐受 (AllergyIntolerance)", "yellow"),
        ("Condition", "🩺 既往/当前诊断 (Condition)", "green"),
        ("Immunization", "💉 疫苗接种记录 (Immunization)", "bright_blue"),
        ("Encounter", "🏨 就诊/入院记录 (Encounter)", "purple"),
        ("CarePlan", "🗂️ 护理/随访计划 (CarePlan)", "grey37"),
        ("CareTeam", "👥 医疗团队 (CareTeam)", "dark_green"),
        ("Claim", "📄 申报/理赔 (Claim)", "dark_orange"),
        ("DocumentReference", "📁 文档引用 (DocumentReference)", "dark_cyan"),
        ("ExplanationOfBenefit", "🧾 给付说明 (ExplanationOfBenefit)", "gold1"),
        ("ImagingStudy", "🖼️ 影像检查 (ImagingStudy)", "magenta"),
        ("Provenance", "📌 溯源信息 (Provenance)", "bright_black"),
        ("Organization", "🏢 机构 (Organization)", "dark_orange"),
        ("Location", "📍 地点 (Location)", "grey54"),
        ("Practitioner", "👩‍⚕️ 医务人员 (Practitioner)", "dark_sea_green"),
        ("PractitionerRole", "🏷️ 职责/角色 (PractitionerRole)", "dark_khaki"),
    ]

    for key, title, color in render_order:
        if counts_by_type.get(key, 0) > 0:
            fn = resource_parsers.get(key)
            try:
                df = fn()
            except Exception as e:
                console.print(f"[red]{key} 解析失败: {e}[/red]")
                console.print(traceback.format_exc())
                continue
            try:
                render_table(title, df, color)
            except Exception as e:
                console.print(f"[red]渲染 {key} 时出错: {e}[/red]")
                console.print(traceback.format_exc())
                sample = [r for r in resources if r.get("resourceType") == key][:3]
                console.print(f"[dim]示例原始资源（最多3条）:[/dim] {sample}")
                continue


if __name__ == "__main__":
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "D:\\Downloads\\fhir\\hospitalInformation1637345232350.json"
    )
    summarize_fhir_json(target)
