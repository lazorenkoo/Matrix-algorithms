folder = fullfile(fileparts(mfilename('fullpath')), 'comparison');
summary = dlmread(fullfile(folder, 'numpy_comparison.csv'), ',', 1, 0);
sizes = summary(:, 1);
csv_file = fopen(fullfile(folder, 'octave_matlab_comparison.csv'), 'w');
text_file = fopen(fullfile(folder, 'octave_matlab_comparison.txt'), 'w');
fprintf(csv_file, 'size,max_absolute_error,matches,time_seconds,repeats,batch_size\n');
fprintf(text_file, 'Porownanie algorytmu Bineta z GNU Octave/MATLAB\n');
fprintf(text_file, 'Wersja programu referencyjnego: %s\n', version);
fprintf(text_file, 'Referencja: A * B, obliczone w programie referencyjnym.\n');
fprintf(text_file, 'Typ danych: double; atol=1e-12, rtol=1e-12\n\n');
fprintf(text_file, 'Czas: mediana czasow serii podzielonych przez liczbe mnozen w serii.\n');
fprintf(text_file, 'Serie zawieraja narzut petli Octave; odczyt i zapis CSV poza pomiarem.\n');
fprintf(text_file, 'Rozmiar | Maks. blad bezwzgledny | Zgodnosc | Czas [s] | Powtorzenia\n');

for index = 1:numel(sizes)
    n = sizes(index);
    size_folder = fullfile(folder, sprintf('size_%d', n));
    A = dlmread(fullfile(size_folder, 'A.csv'), ',');
    B = dlmread(fullfile(size_folder, 'B.csv'), ',');
    C_binet = dlmread(fullfile(size_folder, 'C_binet.csv'), ',');

    if n <= 100
        repeats = 7;
    elseif n <= 200
        repeats = 3;
    else
        repeats = 1;
    end
    C_reference = A * B;
    if n <= 128
        batch_size = 1000;
    elseif n <= 256
        batch_size = 100;
    else
        batch_size = 10;
    end
    times = zeros(repeats, 1);
    for repetition = 1:repeats
        timer = tic;
        for execution = 1:batch_size
            C_reference = A * B;
        end
        times(repetition) = toc(timer) / batch_size;
    end
    elapsed = median(times);
    difference = abs(C_binet - C_reference);
    max_error = max(difference(:));
    tolerance = 1e-12 + 1e-12 * abs(C_reference);
    matching = all(difference(:) <= tolerance(:));

    dlmwrite(fullfile(size_folder, 'C_octave_matlab.csv'), C_reference, ...
        'delimiter', ',', 'precision', '%.17g');
    fprintf(csv_file, '%d,%.17g,%d,%.17g,%d,%d\n', n, max_error, matching, elapsed, repeats, batch_size);
    if matching
        status = 'TAK';
    else
        status = 'NIE';
    end
    fprintf(text_file, '%d x %d | %.6e | %s | %.6e | %d\n', n, n, max_error, status, elapsed, repeats);
    fprintf('n=%d, maksymalny blad=%.6e, zgodnosc=%s, czas=%.6e s\n', n, max_error, status, elapsed);
end

fclose(csv_file);
fclose(text_file);
