function PlotStyle()
%% Set the style for the plots
set(groot, 'defaultFigureColor', [1 1 1]); % White background for pictures
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
% It is not possible to set property defaults that apply to the colorbar label. 
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultColorbarFontSize', 12);
set(groot, 'defaultLegendFontSize', 12);
set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultAxesXGrid', 'on');
set(groot, 'defaultAxesYGrid', 'on');
set(groot, 'defaultAxesZGrid', 'on');
set(groot, 'defaultAxesXMinorTick', 'on');
set(groot, 'defaultAxesYMinorTick', 'on');
set(groot, 'defaultAxesZMinorTick', 'on');
set(groot, 'defaultAxesXMinorGrid', 'on', 'defaultAxesXMinorGridMode', 'manual');
set(groot, 'defaultAxesYMinorGrid', 'on', 'defaultAxesYMinorGridMode', 'manual');
set(groot, 'defaultAxesZMinorGrid', 'on', 'defaultAxesZMinorGridMode', 'manual');
% To see the modified plot style: get(groot, 'default'), to see all the plot style params: get(groot, 'factory')
set(groot,'DefaultFigureColormap',turbo)

%red = [200 36 35]/255; blue = [40 120 181]/255; gray = [200 200 200]/255;
%set(groot,'DefaultFigureColormap',colormap(multigradient([blue;gray;red],'interp','rgb')))

end
