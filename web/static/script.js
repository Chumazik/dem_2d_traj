let currentSimId = 0;
let currentSimParams = null;            // входные параметры текущей симуляции
let currentPartial = null;              // последний снимок /partial_results
let partialPollTimer = null;            // таймер периодического опроса

// Холст: размер больше для комфортного просмотра симуляции
const CANVAS_W = 600;
const CANVAS_H = 600;

function startSim() {
    const mode = document.getElementById('compute_mode').value;  // cpu_jit | cpu | gpu
    const data = {
        num_particles:    document.getElementById('num_particles').value,
        particle_radius:  document.getElementById('particle_radius').value,
        particle_density: document.getElementById('particle_density').value,
        kn:               document.getElementById('kn').value,
        restitution:      document.getElementById('restitution').value,
        friction_static:  document.getElementById('friction_static').value,
        friction_dynamic: document.getElementById('friction_dynamic').value,
        rolling_friction: document.getElementById('rolling_friction').value,
        drum_radius:      document.getElementById('drum_radius').value,
        drum_omega:       document.getElementById('drum_omega').value,
        lifter_height:    document.getElementById('lifter_height').value,
        lifter_width:     document.getElementById('lifter_width').value,
        num_lifters:      document.getElementById('num_lifters').value,
        dt:               document.getElementById('dt').value,
        total_time:       document.getElementById('total_time').value,
        use_jit: mode !== 'cpu',
        use_gpu: mode === 'gpu',
    };
    currentSimId++;
    currentSimParams = data;
    currentComputeMode = mode;
    currentPartial = null;
    fetch('/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...data, sim_id: currentSimId})
    }).then(r => r.json()).then(resp => {
        setComputeModeHint(resp);
        prepareCanvasForLive();
        pollStatus();
    }).catch(err => console.error('Start error:', err));
}

let currentComputeMode = 'cpu_jit';

function setComputeModeHint(resp) {
    const hint = document.getElementById('compute_mode_hint');
    if (!hint) return;
    const label = modeToLabel(currentComputeMode);
    const useJit = resp && resp.use_jit !== undefined ? resp.use_jit : null;
    const useGpu = resp && resp.use_gpu !== undefined ? resp.use_gpu : null;
    const gpuAvail = resp && resp.gpu_available !== undefined ? resp.gpu_available : null;
    let msg = 'Режим: ' + label + ' (use_jit=' + useJit + ', use_gpu=' + useGpu + ')';
    if (currentComputeMode === 'gpu') {
        if (gpuAvail) {
            msg += ' — CuPy доступен, GPU-путь активен.';
        } else {
            msg += ' — CuPy недоступен, автофолбэк на Numba.';
        }
    }
    hint.textContent = msg;
}

function modeToLabel(mode) {
    if (mode === 'gpu')   return 'GPU (CuPy)';
    if (mode === 'cpu')   return 'CPU (чистый Python)';
    return 'CPU (Numba JIT)';
}

function stopSim() {
    fetch('/stop', {method: 'POST'}).catch(err => console.error('Stop error:', err));
}

function pollStatus() {
    fetch('/status').then(r => r.json()).then(st => {
        document.getElementById('prog-val').innerText = st.progress.toFixed(1);
        if (st.running) {
            startPartialPolling();
            setTimeout(pollStatus, 500);
        } else if (st.has_results) {
            stopPartialPolling();
            // финальный кадр — последний известный partial
            drawLiveSnapshot(currentPartial);
            loadResults();
        }
    });
}

// ===== Живой опрос /partial_results =====
function startPartialPolling() {
    if (partialPollTimer) return;
    fetchPartial();
    partialPollTimer = setInterval(fetchPartial, 250);
}

function stopPartialPolling() {
    if (partialPollTimer) {
        clearInterval(partialPollTimer);
        partialPollTimer = null;
    }
}

function fetchPartial() {
    if (!currentSimParams) return;
    const url = '/partial_results?sim_id=' + currentSimId + '&tail=1';
    fetch(url).then(r => r.json()).then(body => {
        if (!body || body.error) return;
        currentPartial = body;
        drawLiveSnapshot(body);
    }).catch(() => { /* игнорируем сетевые ошибки во время опроса */ });
}

function prepareCanvasForLive() {
    const canvas = document.getElementById('preview');
    if (!canvas) return;
    canvas.width = CANVAS_W;
    canvas.height = CANVAS_H;
}

// ===== Отрисовка живого кадра симуляции =====
function drawLiveSnapshot(body) {
    const canvas = document.getElementById('preview');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // фон
    ctx.fillStyle = '#101820';
    ctx.fillRect(0, 0, W, H);

    if (!currentSimParams) return;
    const p = parseSimParams(currentSimParams);
    if (!p || p.drum_radius <= 0) return;

    let timeNow = 0;
    let progress = 0;
    let step = 0;
    let trajectories = [];
    if (body) {
        if (body.time && body.time.length) timeNow = body.time[body.time.length - 1];
        if (body.progress !== undefined) progress = body.progress;
        if (body.step !== undefined) step = body.step;
        if (body.trajectories) trajectories = body.trajectories;
    }

    // Масштаб: в видимую область должны войти барабан + лифтеры + частицы
    const maxExt = p.drum_radius
        + Math.max(p.lifter_height, 0)
        + 2 * p.particle_radius
        + 0.05;
    const scale = Math.min(W, H) / (2 * maxExt);
    const cx = W / 2, cy = H / 2;
    const toPx = m => cx + m * scale;        // смещение по x в пикселях (центрировано)
    const toPy = m => cy - m * scale;        // здесь Y растёт вверх (математические оси)
    const toPx_y = m => cy + m * scale;      // Y растёт вниз (как на канвасе)

    // Барабан (окружность)
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, p.drum_radius * scale, 0, 2 * Math.PI);
    ctx.stroke();
    // Центр
    ctx.fillStyle = '#e0e0e0';
    ctx.beginPath();
    ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
    ctx.fill();

    // Лифтеры, повёрнутые по текущему углу
    if (p.num_lifters > 0 && p.lifter_height > 0 && p.lifter_width > 0) {
        ctx.fillStyle = '#a0522d';
        ctx.strokeStyle = '#5a2d0c';
        ctx.lineWidth = 1.2;
        for (let i = 0; i < p.num_lifters; i++) {
            const baseAngle = (2 * Math.PI * i) / p.num_lifters;
            const ang = baseAngle + p.drum_omega * timeNow;
            const nx = Math.cos(ang);
            const ny = Math.sin(ang);
            // нижняя средняя точка лифтера — на окружности барабана
            const ax = toPx(p.drum_radius * nx);
            const ay = toPx_y(p.drum_radius * ny);
            const halfW = p.lifter_width / 2;
            // Тангенциальное направление: перпендикулярно нормали
            const tx = -ny, ty = nx;
            const x1 = ax + (-halfW) * tx * scale;
            const y1 = ay + (-halfW) * ty * scale;
            const x2 = ax + ( halfW) * tx * scale;
            const y2 = ay + ( halfW) * ty * scale;
            const x3 = x2 + p.lifter_height * nx * scale;
            const y3 = y2 + p.lifter_height * ny * scale;
            const x4 = x1 + p.lifter_height * nx * scale;
            const y4 = y1 + p.lifter_height * ny * scale;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x3, y3);
            ctx.lineTo(x4, y4);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
    }

    // Частицы: рисуем последнюю зафиксированную позицию
    const pr = Math.max(1, p.particle_radius * scale);
    ctx.fillStyle = 'rgba(120, 180, 235, 0.95)';
    ctx.strokeStyle = '#1c3f5a';
    ctx.lineWidth = 0.8;
    for (let i = 0; i < trajectories.length; i++) {
        const traj = trajectories[i];
        if (!traj || traj.length === 0) continue;
        const last = traj[traj.length - 1];
        const px = toPx(last[0]);
        const py = toPx_y(last[1]);
        ctx.beginPath();
        ctx.arc(px, py, pr, 0, 2 * Math.PI);
        ctx.fill();
        if (pr >= 3) {
            ctx.stroke();
        }
    }

    // Оверлей статуса
    drawStatusOverlay(ctx, W, H, {
        step, progress, t: timeNow, nPart: trajectories.length,
    });
}

function drawStatusOverlay(ctx, W, H, info) {
    const lines = [
        'Шаг: ' + info.step,
        'Время: ' + info.t.toFixed(4) + ' с',
        'Прогресс: ' + info.progress.toFixed(1) + ' %',
        'Частиц: ' + info.nPart,
    ];
    const w = 200, h = 18 + lines.length * 16;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(12, 12, w, h);
    ctx.strokeStyle = '#999';
    ctx.strokeRect(12, 12, w, h);
    ctx.fillStyle = '#f0f0f0';
    ctx.font = '12px Consolas, monospace';
    for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], 22, 28 + i * 16);
    }
}

function parseSimParams(p) {
    return {
        num_particles:   Math.max(0, Math.floor(parseFloat(p.num_particles) || 0)),
        particle_radius: parseFloat(p.particle_radius) || 0,
        particle_density: parseFloat(p.particle_density) || 0,
        drum_radius:     parseFloat(p.drum_radius) || 0,
        drum_omega:      parseFloat(p.drum_omega) || 0,
        lifter_height:   parseFloat(p.lifter_height) || 0,
        lifter_width:    parseFloat(p.lifter_width) || 0,
        num_lifters:     Math.max(0, Math.floor(parseFloat(p.num_lifters) || 0)),
    };
}

function loadResults() {
    fetch('/results').then(r => r.json()).then(res => {
        if (res.error) {
            console.error('Results error:', res.error);
            return;
        }
        console.log('Trajectories received:', res.trajectories);
        // Траектории
        if (!res.trajectories || res.trajectories.length === 0) {
            console.warn('Пустые траектории, рисуем пустой график');
            Plotly.newPlot('traj-plot', [], {title: 'Траектории частиц (нет данных)'});
        } else {
            const trajTraces = res.trajectories.map((traj, idx) => ({
                x: traj.map(p => p[0]),
                y: traj.map(p => p[1]),
                mode: 'lines',
                name: 'Частица ' + idx,
                line: {width: 1}
            }));
            Plotly.newPlot('traj-plot', trajTraces, {title: 'Траектории частиц'});
        }

        // Момент
        if (res.time && res.torque_history && res.time.length > 0) {
            Plotly.newPlot('torque-plot', [{
                x: res.time,
                y: res.torque_history,
                mode: 'lines',
                line: {color: 'red'}
            }], {title: 'Приводной момент во времени'});
        } else {
            Plotly.newPlot('torque-plot', [], {title: 'Приводной момент (нет данных)'});
        }

        // Финальный кадр живой отрисовки — последний сохранённый partial либо последние
        // точки полной траектории
        if (res.trajectories && res.trajectories.length > 0 && currentSimParams) {
            drawLiveSnapshot({
                time: res.time && res.time.length ? [res.time[res.time.length - 1]] : [],
                progress: 100,
                step: res.trajectories[0].length,
                trajectories: res.trajectories.map(t => t.length ? [t[t.length - 1]] : []),
            });
        }
    }).catch(err => console.error('Load results error:', err));
}

// ===== Статичный предпросмотр (до старта симуляции) =====
const PREVIEW_IDS = [
    'num_particles', 'particle_radius',
    'drum_radius', 'lifter_height', 'lifter_width', 'num_lifters'
];

function readPreviewParams() {
    return parseSimParams({
        num_particles:  document.getElementById('num_particles').value,
        particle_radius: document.getElementById('particle_radius').value,
        drum_radius:    document.getElementById('drum_radius').value,
        lifter_height:  document.getElementById('lifter_height').value,
        lifter_width:   document.getElementById('lifter_width').value,
        num_lifters:    document.getElementById('num_lifters').value,
    });
}

function updatePreview() {
    const canvas = document.getElementById('preview');
    if (!canvas) return;
    if (canvas.width !== 500 || canvas.height !== 500) {
        canvas.width = 500;
        canvas.height = 500;
    }
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, W, H);

    const p = readPreviewParams();
    if (p.drum_radius <= 0) return;

    const maxExtent = p.drum_radius + Math.max(p.lifter_height, 0) + p.drum_radius * 0.2;
    const scale = Math.min(W, H) / (2 * maxExtent);
    const cx = W / 2, cy = H / 2;
    const toPx = (m) => m * scale;

    // Барабан
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, toPx(p.drum_radius), 0, Math.PI * 2);
    ctx.stroke();

    // Лифтеры (статичные — без учёта вращения)
    if (p.num_lifters > 0 && p.lifter_height > 0 && p.lifter_width > 0) {
        ctx.fillStyle = '#a0522d';
        ctx.strokeStyle = '#5a2d0c';
        const baseRadius = p.drum_radius;
        for (let i = 0; i < p.num_lifters; i++) {
            const baseAngle = (2 * Math.PI * i) / p.num_lifters - Math.PI / 2;
            const nx = Math.cos(baseAngle);
            const ny = Math.sin(baseAngle);
            const ax = cx + toPx(baseRadius * nx);
            const ay = cy + toPx(baseRadius * ny);
            const halfW = p.lifter_width / 2;
            const x1 = ax + toPx(-halfW * (-ny));
            const y1 = ay + toPx(-halfW * (nx));
            const x2 = ax + toPx( halfW * (-ny));
            const y2 = ay + toPx( halfW * (nx));
            const x3 = x2 + toPx(p.lifter_height * nx);
            const y3 = y2 + toPx(p.lifter_height * ny);
            const x4 = x1 + toPx(p.lifter_height * nx);
            const y4 = y1 + toPx(p.lifter_height * ny);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineTo(x3, y3);
            ctx.lineTo(x4, y4);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
    }

    // Частицы — простая гексагональная упаковка внутри эффективной области
    if (p.num_particles > 0 && p.particle_radius > 0) {
        const effectiveR = Math.max(0, p.drum_radius - 2 * p.particle_radius);
        if (effectiveR > 0) {
            const r = p.particle_radius;
            const spacing = 2 * r;
            const rowH = spacing * Math.sqrt(3) / 2;
            ctx.fillStyle = 'rgba(70, 130, 180, 0.85)';
            ctx.strokeStyle = '#1c3f5a';
            let placed = 0;
            for (let y = -effectiveR; y <= effectiveR && placed < p.num_particles; y += rowH) {
                const rowOffset = (Math.round(y / rowH) % 2 === 0) ? 0 : r;
                for (let x = -effectiveR - rowOffset; x <= effectiveR && placed < p.num_particles; x += spacing) {
                    if (x * x + y * y > effectiveR * effectiveR) continue;
                    ctx.beginPath();
                    ctx.arc(cx + toPx(x + rowOffset), cy + toPx(y), Math.max(1, toPx(r)), 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                    placed++;
                }
            }
        }
    }
}

// Инициализация превью и подписка на изменения
document.addEventListener('DOMContentLoaded', () => {
    PREVIEW_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', updatePreview);
            el.addEventListener('change', updatePreview);
        }
    });
    updatePreview();
    setupTabs();
});

// ====== Tabs ======
function setupTabs() {
    document.querySelectorAll('.tablinks').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            document.querySelectorAll('.tablinks').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tabcontent').forEach(c => c.classList.add('hidden'));
            const target = document.getElementById(tabId);
            if (target) target.classList.remove('hidden');
        });
    });
}

// ====== Analytical tab ======
const ANALYTICAL_FIELDS = [
    {label: 'Эффективный диаметр (ft)',            key: 'mill_diameter_ft',          un: 'ft'},
    {label: 'Радиус мельницы (ft)',                key: 'mill_radius_ft',            un: 'ft'},
    {label: 'Длина окружности (ft)',               key: 'mill_circumference_ft',      un: 'ft'},
    {label: 'Площадь сечения (ft²)',               key: 'mill_cross_section_ft2',     un: 'ft²'},
    {label: 'Диаметр шара (ft)',                   key: 'ball_diameter_ft',           un: 'ft'},
    {label: 'Объём шара (in³)',                    key: 'ball_volume_in3',            un: 'in³'},
    {label: 'Масса шара (lbm)',                    key: 'ball_mass_lbm',              un: 'lbm'},
    {label: 'Масса шара (kg)',                     key: 'ball_mass_kg',               un: 'kg'},
    {label: 'Радиус центра шара r (ft)',           key: 'effective_ball_radius_ft',   un: 'ft'},
    {label: 'Высота лифтера (ft)',                 key: 'lifter_height_ft',           un: 'ft'},
    {label: 'Критическая скорость (rpm)',          key: 'critical_speed_rpm',         un: 'rpm'},
    {label: 'Критическая скорость (rad/s)',        key: 'critical_speed_rad_s',       un: 'rad/s'},
    {label: 'Рабочая скорость (rpm)',              key: 'operating_speed_rpm',        un: 'rpm'},
    {label: 'Рабочая скорость (rad/s)',            key: 'operating_speed_rad_s',      un: 'rad/s'},
    {label: 'Доля критической',                    key: 'fraction_of_critical',       un: ''},
    {label: 'Линейная скорость барабана (ft/s)',   key: 'mill_peripheral_speed_ft_s', un: 'ft/s'},
    {label: 'Линейная скорость шара (ft/s)',       key: 'ball_peripheral_speed_ft_s', un: 'ft/s'},
    {label: 'Центрострем. на r (в долях g)',       key: 'centrifugal_at_r_g',         un: 'g'},
    {label: 'Угол shoulder (°)',                   key: 'shoulder_angle_deg',         un: '°'},
    {label: 'Позиция shoulder X (ft)',             key: 'shoulder_position_x_ft',     un: 'ft'},
    {label: 'Позиция shoulder Y (ft)',             key: 'shoulder_position_y_ft',     un: 'ft'},
    {label: 'Shoulder clock',                      key: 'shoulder_clock_hours',       un: 'ч'},
    {label: 'Стартовая скорость Vx (ft/s)',        key: 'launch_velocity_x_ft_s',     un: 'ft/s'},
    {label: 'Стартовая скорость Vy (ft/s)',        key: 'launch_velocity_y_ft_s',     un: 'ft/s'},
    {label: 'Стартовая |V| (ft/s)',                key: 'launch_speed_ft_s',          un: 'ft/s'},
    {label: 'Время апекса (s)',                    key: 'apex_time_s',                un: 's'},
    {label: 'Высота апекса (ft)',                  key: 'apex_height_ft',             un: 'ft'},
    {label: 'X в апексе (ft)',                     key: 'apex_position_x_ft',         un: 'ft'},
    {label: 'Время полёта (s)',                    key: 'flight_time_s',              un: 's'},
    {label: 'Точка удара X (ft)',                  key: 'impact_position_x_ft',       un: 'ft'},
    {label: 'Точка удара Y (ft)',                  key: 'impact_position_y_ft',       un: 'ft'},
    {label: 'Расст. до центра при ударе (ft)',     key: 'impact_distance_from_center_ft', un: 'ft'},
    {label: 'Vx в момент удара (ft/s)',            key: 'impact_velocity_x_ft_s',     un: 'ft/s'},
    {label: 'Vy в момент удара (ft/s)',            key: 'impact_velocity_y_ft_s',     un: 'ft/s'},
    {label: '|V| в момент удара (ft/s)',           key: 'impact_speed_ft_s',          un: 'ft/s'},
    {label: 'Угол точки удара от −y (°)',          key: 'impact_angle_from_vertical_deg', un: '°'},
    {label: 'Угол V к касательной (°)',            key: 'impact_angle_from_tangent_deg', un: '°'},
    {label: 'Кинетическая энергия (ft·lbf)',       key: 'impact_kinetic_energy_ftlbf', un: 'ft·lbf'},
    {label: 'Кинетическая энергия (J)',            key: 'impact_kinetic_energy_joules', un: 'J'},
    {label: 'Доля заполнения',                     key: 'filling_fraction',           un: ''},
    {label: 'Kidney angle (°)',                    key: 'kidney_angle_deg',           un: '°'},
    {label: 'Toe angle (°)',                       key: 'toe_angle_deg',              un: '°'},
    {label: 'Toe clock',                           key: 'toe_clock_hours',            un: 'ч'},
    {label: 'Объём заряда (ft³)',                  key: 'charge_volume_ft3',          un: 'ft³'},
    {label: 'Масса заряда (lbm)',                  key: 'charge_mass_lbm',            un: 'lbm'},
    {label: 'Масса заряда (kg)',                   key: 'charge_mass_kg',             un: 'kg'},
    {label: 'Σ кин. энергия заряда (J)',           key: 'charge_kinetic_energy_joules', un: 'J'},
    {label: 'Угол разгрузки (°)',                  key: 'discharge_angle_deg',        un: '°'},
    {label: 'Момент от заряда (ft·lbf)',           key: 'torque_due_to_charge_ftlbf', un: 'ft·lbf'},
    {label: 'Момент от заряда (N·m)',              key: 'torque_due_to_charge_nm',    un: 'N·m'},
    {label: 'Мощность (hp)',                       key: 'power_draw_hp',              un: 'hp'},
    {label: 'Мощность (kW)',                       key: 'power_draw_kw',              un: 'kW'},
];

function readAnalyticalInputs() {
    const v = id => parseFloat(document.getElementById(id).value);
    return {
        effective_mill_diameter_ft: v('a_d_eff')         || 36.0,
        ball_diameter_in:           v('a_d_ball')        || 5.0,
        static_friction:            v('a_mu_s')          || 0.05,
        dynamic_friction:           v('a_mu_d')          || 0.20,
        lifter_face_angle_deg:      v('a_lifter_angle')  || 15,
        lifter_height_in:           v('a_lifter_height') || 8,
        pct_critical_speed:         v('a_pct_cs')        || 76,
        apparent_mill_filling:      v('a_filling')       || 28,
        angle_of_repose_deg:        v('a_repose')        || 35,
        ball_density_lb_in3:        v('a_ball_density')  || 0.284,
        n_traj_points:              41,
    };
}

function fmtVal(v) {
    if (typeof v !== 'number' || isNaN(v)) return String(v);
    if (Math.abs(v) >= 1e4 || (Math.abs(v) > 0 && Math.abs(v) < 1e-3)) {
        return v.toExponential(4);
    }
    return v.toFixed(4);
}

function renderAnalyticalTable(out) {
    const tbl = document.getElementById('analytical-table');
    let html = '<tr><th>Показатель</th><th>Значение</th><th>Ед.</th></tr>';
    ANALYTICAL_FIELDS.forEach(f => {
        const v = out[f.key];
        if (v === undefined || v === null) return;
        html += `<tr><td>${f.label}</td><td class="num">${fmtVal(v)}</td><td>${f.un}</td></tr>`;
    });
    tbl.innerHTML = html;

    const diff = out.diff_vs_molycop || {};
    const diffTbl = document.getElementById('diff-table');
    let dhtml = '<tr><th>Показатель</th><th>Аналитика</th><th>Moly-Cop</th><th>Δ</th></tr>';
    for (const k of Object.keys(diff)) {
        const r = diff[k];
        if (r.sheet === undefined) continue;
        const dv = (typeof r.analytic === 'number' && typeof r.sheet === 'number')
            ? (r.analytic - r.sheet) : '—';
        const dvStr = (typeof dv === 'number') ? fmtVal(dv) : '—';
        dhtml += `<tr><td>${k}</td><td class="num">${fmtVal(r.analytic)}</td><td class="num">${fmtVal(r.sheet)}</td><td class="num">${dvStr}</td></tr>`;
    }
    diffTbl.innerHTML = dhtml;
}

function renderAnalyticalPlot(out) {
    const R = out.mill_radius_ft;
    const cx = 0, cy = 0;
    const thetaN = 200;
    const circleX = [], circleY = [];
    for (let i = 0; i <= thetaN; i++) {
        const t = 2 * Math.PI * i / thetaN;
        circleX.push(cx + R * Math.cos(t));
        circleY.push(cy + R * Math.sin(t));
    }
    const traj = out.trajectory;
    const trajX = traj.map(p => p.x), trajY = traj.map(p => p.y);
    const impact = {x: out.impact_position_x_ft, y: out.impact_position_y_ft};
    const shoulder = {x: out.shoulder_position_x_ft, y: out.shoulder_position_y_ft};

    const traces = [
        {
            x: circleX, y: circleY, mode: 'lines',
            name: 'Барабан', line: {color: '#888', width: 2},
            hoverinfo: 'name'
        },
        {
            x: trajX, y: trajY, mode: 'lines+markers',
            name: 'Траектория полёта',
            line: {color: '#c0392b', width: 2},
            marker: {size: 4, color: '#c0392b'},
        },
        {
            x: [shoulder.x], y: [shoulder.y], mode: 'markers',
            name: 'Shoulder (отрыв)', marker: {color: '#2980b9', size: 10, symbol: 'circle'},
        },
        {
            x: [impact.x], y: [impact.y], mode: 'markers',
            name: 'Impact (удар)', marker: {color: '#27ae60', size: 12, symbol: 'x'},
        },
    ];

    const layout = {
        title: 'Траектория мелющего тела (аналитика)',
        xaxis: {
            title: 'X (ft)', zeroline: true,
            range: [-R * 1.2, R * 1.2], scaleanchor: 'y', scaleratio: 1,
        },
        yaxis: {title: 'Y (ft)', zeroline: true, range: [-R * 1.2, R * 1.2]},
        shapes: [{
            type: 'circle', xref: 'x', yref: 'y',
            x0: cx - R, x1: cx + R, y0: cy - R, y1: cy + R,
            line: {color: '#888', width: 1.5},
        }],
        showlegend: true, margin: {l: 50, r: 20, t: 50, b: 50},
    };
    Plotly.newPlot('analytical-plot', traces, layout, {responsive: true});
}

function runAnalytical() {
    const payload = readAnalyticalInputs();
    fetch('/analytical', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
    }).then(r => r.json()).then(body => {
        if (body.error) {
            alert('Ошибка: ' + body.error);
            return;
        }
        renderAnalyticalTable(body);
        renderAnalyticalPlot(body);
    }).catch(err => alert('Ошибка сети: ' + err));
}
