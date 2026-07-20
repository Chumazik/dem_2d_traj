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
    fetch('/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    }).then(r => r.json()).then(() => {
        pollStatus();
    });
}

function pollStatus() {
    fetch('/status').then(r => r.json()).then(st => {
        document.getElementById('prog-val').innerText = st.progress.toFixed(1);
        if (st.running) {
            setTimeout(pollStatus, 200);
        } else if (st.has_results) {
            loadResults();
        }
    });
}

function loadResults() {
    fetch('/results').then(r => r.json()).then(res => {
        // Траектории
        const trajTraces = res.trajectories.map(traj => ({
            x: traj.map(p => p[0]),
            y: traj.map(p => p[1]),
            mode: 'lines',
            line: {width: 1}
        }));
        Plotly.newPlot('traj-plot', trajTraces, {title: 'Траектории частиц'});

        // Момент
        Plotly.newPlot('torque-plot', [{
            x: res.time,
            y: res.torque_history,
            mode: 'lines',
            line: {color: 'red'}
        }], {title: 'Приводной момент во времени'});
    });
}
