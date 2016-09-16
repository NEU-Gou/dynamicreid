function [region_idx, BBox, region_mask] = GenerateGridBBox_numP( imsz, nr, nc, overlap )
%GENERATEGRIDBBOX_NUMP Generate 50% overlapped grid boxes given # of girds
% [region_idx, BBox, region_mask] = GenerateGridBBox_numP( imsz, nr, nc )
% INPUT   --- imsz[y,x] image size
%             nr # of patches along the row
%             nc # of patches along the col
% OUTPUT  --- regi
% Write by Mengran Gou @ 10/05/2015 
if overlap
    % calculate patch size (50% overlapped along each side)
    szPx = 2*(imsz(2)/(nc+1));
    szPy = 2*(imsz(1)/(nr+1));
    [region_idx, BBox, region_mask] =GenerateGridBBox(imsz, [szPy szPx], [szPy szPx]/2);
else 
    szPx = imsz(2)/nc;
    szPy = imsz(1)/nr;
    [region_idx, BBox, region_mask] =GenerateGridBBox(imsz, [szPy szPx], [szPy szPx]);
end


% region_idx = cell2mat(region_idx')';

end

