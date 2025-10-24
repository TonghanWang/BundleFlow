import torch
import plotly.figure_factory as ff
import plotly.graph_objects as go


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Input #{hex_color} is not in #RRGGBB format")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


@torch.no_grad()
def draw_flow(rectified_flow, z0, weights, N=None, file_name=None, x_dim=0, y_dim=1, scale=1):
    xylim = 1.5
    traj, _, _ = rectified_flow.sample_ode(z0=z0, N=N)

    # Set up the figure
    fig = go.Figure()

    # Define color palette
    colors = [
        '#FA7F6F',  # Example colors
        '#6571F2',
        '#489893',
        '#90A4AE',
        '#5CC99A',
        '#63D0EF',
    ]
    marker_size = 6 * scale
    linewidth = 3 * scale
    opa_now = 0.1
    line_number = 1  # Number of arrows

    traj_particles = torch.stack(traj)

    for i in range(traj_particles.shape[1]):
        weight = weights[i]
        if weight < 1e-4:
            weight = 0
        traj_particles_i = traj_particles[:, i]
        differences = traj_particles_i[1:] - traj_particles_i[:-1]
        quiver = ff.create_quiver(
            traj_particles_i[:-1, x_dim], traj_particles_i[:-1, y_dim],
            differences[:, x_dim], differences[:, y_dim],
            scale=1.0,
            arrow_scale=0.01,  # Adjust as needed
            line_color=hex_to_rgba(colors[5], weight),
            # Remove or adjust the opacity parameter since transparency is handled by RGBA
            # opacity=0.4,
            line=dict(width=linewidth, ),
        )

        fig.add_trace(quiver.data[0])

    fig.add_trace(go.Scatter(
        x=traj[0][:, x_dim].cpu().numpy(), y=traj[0][:, y_dim].cpu().numpy(),
        name='denoised state',
        mode='markers',
        marker=dict(
            size=marker_size,  # Adjust as needed
            color=colors[4],  # Use original color for markers
            # Optionally, add transparency to markers as well
            opacity=weights,
            symbol='circle'
        ),
    ))

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=traj[-1][:, x_dim].cpu().numpy(), y=traj[-1][:, y_dim].cpu().numpy(),
        name='denoised state',
        mode='markers',
        marker=dict(
            size=marker_size,  # Adjust as needed
            color=colors[1],  # Use original color for markers
            # Optionally, add transparency to markers as well
            opacity=weights,
            symbol='circle-open'
        ),
    ))

    # Update layout for better appearance
    fig.update_layout(
        width=800,
        height=800,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        # title='Distribution and Transport Trajectory',
    )

    # Set axis properties
    fig.update_xaxes(range=[-0.1, 1.1], tickfont=dict(size=18))
    fig.update_yaxes(range=[-0.1, 1.1], tickfont=dict(size=18))

    if file_name is None:
        # Show the plot
        fig.show()
    else:
        fig.write_image(file_name)


@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None, file_name=None):
    xylim = 1.5
    traj, _, _ = rectified_flow.sample_ode(z0=z0, N=N)

    # Set up the figure
    fig = go.Figure()

    # Define color palette
    colors = [
        '#FA7F6F',  # Example colors
        '#6571F2',
        '#489893',
        '#90A4AE',
        '#5CC99A',
        '#63D0EF',
    ]
    marker_size = 6
    linewidth = 3
    opa_now = 0.1
    line_number = 1  # Number of arrows

    traj_particles = torch.stack(traj)

    for i in range(traj_particles.shape[1]):
        traj_particles_i = traj_particles[:, i]
        differences = traj_particles_i[1:] - traj_particles_i[:-1]
        quiver = ff.create_quiver(
            traj_particles_i[:-1, 0], traj_particles_i[:-1, 1],
            differences[:, 0], differences[:, 1],
            scale=1.0,
            arrow_scale=0.01,  # Adjust as needed
            line_color=hex_to_rgba(colors[5], opa_now),
            # Remove or adjust the opacity parameter since transparency is handled by RGBA
            # opacity=0.4,
            line=dict(width=linewidth, ),
        )

        fig.add_trace(quiver.data[0])

    fig.add_trace(go.Scatter(
        x=traj[0][:, 0].cpu().numpy(), y=traj[0][:, 1].cpu().numpy(),
        name='denoised state',
        mode='markers',
        marker=dict(
            size=marker_size,  # Adjust as needed
            color=colors[4],  # Use original color for markers
            # Optionally, add transparency to markers as well
            opacity=0.2,
            symbol='circle'
        ),
    ))

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=traj[-1][:, 0].cpu().numpy(), y=traj[-1][:, 1].cpu().numpy(),
        name='denoised state',
        mode='markers',
        marker=dict(
            size=marker_size,  # Adjust as needed
            color=colors[1],  # Use original color for markers
            # Optionally, add transparency to markers as well
            opacity=0.2,
            symbol='circle-open'
        ),
    ))

    # Update layout for better appearance
    fig.update_layout(
        width=600,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )


    # Set axis properties
    fig.update_xaxes(range=[-0.1, 1.1], tickfont=dict(size=18))
    fig.update_yaxes(range=[-0.1, 1.1], tickfont=dict(size=18))

    # Show the plot
    if file_name is None:
        fig.show()
    else:
        fig.write_image(file_name)


if __name__ == '__main__':
    import numpy as np

    # Define the two arrays
    array1 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])  # First column
    array2 = np.array([0.549, 0.875, 1.219, 1.576, 1.943, 2.318, 2.699, 3.086, 3.478])  # Second column

    print(array2 / array1)
