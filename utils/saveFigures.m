function saveFigures(fName, pos)
    set(gcf, 'PaperPositionMode', 'auto');
    if nargin > 1
        set(gcf, 'pos', pos);
    end
    [~, ~, ext] = fileparts(fName);
    if strcmpi(ext, '.eps')
        dr = '-depsc2';
    elseif strcmpi(ext, '.jpg')
        dr = '-djpeg';
    else
        dr = ['-d', ext(2:end)];
    end
    print(gcf, dr, '-noui', '-loose', fName);
end