let currentSimId = 0;

function startSim() {
    const data = {
        num_particles: document.getElementById('num_particles').value,
        particle_radius: document.getElementById('particle_radius').value,
        particle_density: document.getElementById('particle_density').value,
        kn: document.getElementById('kn').value,
        restitution: document.getElementById('restitution').value,
        friction_static: document.getElementById('friction_static').value,
        friction_dynamic: document.getElementById('friction_dynamic').value,
        rolling_friction: document.getElementById('rolling_friction').value,
        drum_radius: document.getElementById('drum_radius').value,
        drum_omega: document.getElementById('drum_omega').value,
        lifter_height: document.getElementById('lifter_height').value,
        lifter_width: document.getElementById('lifter_width').value,
        num_lifters: document.getElementById('num_lifters').value,
        dt: document.getElementById('dt').value,
        total_time: document.getElementById('total_time').value
    };
    currentSimId++;
    fetch('/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({...data, sim_id: currentSimId})
    }).then(r => r.json()).then(() => {
        pollStatus();
    }).catch(err => console.error('Start error:', err));
}

function stopSim() {
    fetch('/stop', {method: 'POST'}).catch(err => console.error('Stop error:', err));
}

function pollStatus() {
    fetch('/status').then(r => r.json()).then(st => {
        document.getElementById('prog-val').innerText = st.progress.toFixed(1);
        if (st.running) {
            setTimeout(pollStatus, 500);
        } else if (st.has_results) {
            loadResults();
        }
    }).catch(err => console.error('Status error:', err));
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
    }).catch(err => console.error('Load results error:', err));
}

// ====== Предпросмотр геометрии ======
const PREVIEW_IDS = [
    'num_particles', 'particle_radius',
    'drum_radius', 'lifter_height', 'lifter_width', 'num_lifters'
];

function readPreviewParams() {
    const v = id => parseFloat(document.getElementById(id).value);
    return {
        num_particles: Math.max(0, Math.floor(v('num_particles') || 0)),
        particle_radius: v('particle_radius') || 0,
        drum_radius: v('drum_radius') || 0,
        lifter_height: v('lifter_height') || 0,
        lifter_width: v('lifter_width') || 0,
        num_lifters: Math.max(0, Math.floor(v('num_lifters') || 0))
    };
}

function updatePreview() {
    const canvas = document.getElementById('preview');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Очистка
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, W, H);

    const p = readPreviewParams();
    if (p.drum_radius <= 0) {
        return;
    }

    // Масштаб: помещаем барабан + лифтеры в канвас с отступом
    const maxExtent = p.drum_radius + Math.max(p.lifter_height, 0) + p.drum_radius * 0.2;
    const scale = Math.min(W, H) / (2 * maxExtent);
    const cx = W / 2;
    const cy = H / 2;
    const toPx = (m) => m * scale;

    // Барабан
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, toPx(p.drum_radius), 0, Math.PI * 2);
    ctx.stroke();

    // Лифтеры
    if (p.num_lifters > 0 && p.lifter_height > 0 && p.lifter_width > 0) {
        ctx.fillStyle = '#a0522d';
        ctx.strokeStyle = '#5a2d0c';
        const baseRadius = p.drum_radius;
        for (let i = 0; i < p.num_lifters; i++) {
            const baseAngle = (2 * Math.PI * i) / p.num_lifters - Math.PI / 2;
            // Центр нижней грани лифтера на окружности
            const ax = cx + toPx(baseRadius * Math.cos(baseAngle));
            const ay = cy + toPx(baseRadius * Math.sin(baseAngle));
            // Вектор наружу
            const nx = Math.cos(baseAngle);
            const ny = Math.sin(baseAngle);
            // Точки прямоугольника (4 угла)
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
});
